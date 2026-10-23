"""Official ChatGPT-plan OAuth: dynamic OSS registration, PKCE, OIDC and refresh."""
from __future__ import annotations
import base64, hashlib, json, os, secrets, tempfile, threading, time, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
import jwt
import requests

AUTH = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"

class AuthenticationError(RuntimeError):
    pass

class CredentialStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("RAG_AUTH_DIR", "~/.config/rag-pipeline-lab")).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    def _write(self, path, payload):
        fd, tmp = tempfile.mkstemp(prefix=".oauth-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            if os.name != "nt": os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def host_id(self):
        path = self.root / "host.json"
        if not path.exists():
            self._write(path, {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())})
        return json.loads(path.read_text())["ext_agent_host_id"]

    def load(self):
        path = self.root / "session.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, data):
        self._write(self.root / "session.json", data)


class OAuthSession:
    def __init__(self, store=None, http=None):
        self.store, self.http = store or CredentialStore(), http or requests.Session()
        self._lock = threading.Lock()

    def _exchange(self, fields):
        r = self.http.post(AUTH + "/api/accounts/oauth/token", data=fields, timeout=30)
        if not r.ok:
            raise AuthenticationError("OAuth exchange failed (HTTP %s). Reauthorize." % r.status_code)
        return r.json()

    def _validate_id_token(self, token, client_id, nonce=None):
        discovery = self.http.get(AUTH + "/.well-known/openid-configuration", timeout=20)
        discovery.raise_for_status()
        settings = discovery.json()
        signing_key = jwt.PyJWKClient(settings["jwks_uri"]).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, signing_key.key, algorithms=["RS256"], audience=client_id,
                            issuer=settings["issuer"], options={"require":["exp","iss","aud","sub"]})
        if nonce is not None and claims.get("nonce") != nonce:
            raise AuthenticationError("OIDC nonce mismatch")
        return claims

    def login(self, *, new_account=False, open_browser=True, timeout=180):
        saved = None if new_account else self.store.load()
        host = self.store.host_id()
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        callback = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlsplit(self.path)
                if url.path != "/auth/callback":
                    self.send_error(404)
                    return
                callback.update({k:v[0] for k,v in parse_qs(url.query).items() if v})
                valid = secrets.compare_digest(callback.get("state", ""), state)
                self.send_response(200 if valid else 400)
                self.end_headers()
                self.wfile.write(b"Authorization received. Return to terminal." if valid else b"Invalid state.")
            def log_message(self, *args):
                pass  # Never log callback codes or ID token hints.

        server = HTTPServer(("127.0.0.1", 0), Handler)
        server.timeout = timeout
        redirect = "http://127.0.0.1:%d/auth/callback" % server.server_port
        client_id = saved["client_id"] if saved else "dynamic_agent_client"
        params = dict(client_id=client_id, ext_agent_host_id=host,
                      response_type="code", redirect_uri=redirect, scope=SCOPES,
                      resource=RESOURCE, state=state, nonce=nonce,
                      code_challenge_method="S256", code_challenge=challenge)
        if saved:
            if saved.get("id_token"): params["id_token_hint"] = saved["id_token"]
            if saved.get("email"): params["login_hint"] = saved["email"]
        else:
            params["agent_name_hint"] = "RAG Pipeline Lab"
        url = AUTH + "/api/accounts/authorize?" + urlencode(params)
        try:
            if open_browser: webbrowser.open(url)
            else:
                print("Open locally; do not share or log this URL:\n" + url)
            server.handle_request()
        finally:
            server.server_close()
        if not secrets.compare_digest(callback.get("state", ""), state):
            raise AuthenticationError("Missing callback or OAuth state mismatch")
        if callback.get("error"):
            raise AuthenticationError("Authorization declined: " + callback["error"])
        issued = callback.get("client_id", client_id)
        if (not saved and issued == "dynamic_agent_client") or (saved and issued != client_id):
            raise AuthenticationError("Missing or inconsistent issued client ID")
        if not callback.get("code"):
            raise AuthenticationError("No authorization code received")
        token = self._exchange(dict(grant_type="authorization_code", client_id=issued,
                                    code=callback["code"], code_verifier=verifier,
                                    redirect_uri=redirect, resource=RESOURCE))
        claims = self._validate_id_token(token["id_token"], issued, nonce)
        if saved and saved["subject"] != claims["sub"]:
            raise AuthenticationError("Returning session changed account identity")
        scopes = token.get("scope","").split()
        if "chatgpt.tokens.use.direct" not in scopes:
            raise AuthenticationError("ChatGPT plan permission not granted")
        data = dict(client_id=issued, ext_agent_host_id=host, subject=claims["sub"],
                    email=claims.get("email"), id_token=token["id_token"],
                    access_token=token["access_token"], refresh_token=token.get("refresh_token"),
                    scopes=scopes, expires_at=time.time()+token["expires_in"])
        self.store.save(data)
        return {"email": data["email"], "client_id": issued, "scopes": scopes}

    def token(self):
        with self._lock:
            saved = self.store.load()
            if not saved:
                raise AuthenticationError("Run: python -m rag_lab.llms.cli login")
            if "chatgpt.tokens.use.direct" not in saved.get("scopes", []):
                raise AuthenticationError("ChatGPT plan usage was not authorized")
            if saved["expires_at"] <= time.time()+120:
                if not saved.get("refresh_token"):
                    raise AuthenticationError("Sign in again to refresh the session")
                new = self._exchange(dict(grant_type="refresh_token",
                                          client_id=saved["client_id"],
                                          refresh_token=saved["refresh_token"], resource=RESOURCE))
                if "id_token" in new:
                    identity = self._validate_id_token(new["id_token"], saved["client_id"])
                    if identity["sub"] != saved["subject"]:
                        raise AuthenticationError("Refresh changed account identity")
                scopes = new.get("scope", " ".join(saved["scopes"])).split()
                if "chatgpt.tokens.use.direct" not in scopes:
                    raise AuthenticationError("Refresh lost plan usage permission")
                saved.update(access_token=new["access_token"],
                             refresh_token=new.get("refresh_token", saved["refresh_token"]),
                             id_token=new.get("id_token", saved["id_token"]),
                             scopes=scopes, expires_at=time.time()+new["expires_in"])
                self.store.save(saved)
            return saved["access_token"]

    def models(self):
        r = self.http.get(RESOURCE+"/models", headers={"Authorization":"Bearer "+self.token()}, timeout=30)
        r.raise_for_status()
        return [{"slug":m["slug"], "display_name":m.get("display_name",m["slug"])}
                for m in r.json().get("models",[])
                if m.get("visibility")=="list" and m.get("slug")]

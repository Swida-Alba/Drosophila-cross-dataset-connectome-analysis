import os

class TokenManager:
    """
    Helper class for managing authentication tokens (NEUPRINT_TOKEN, CAVE_TOKEN, BANC_TOKEN).

    Token Priority (in order of precedence):
    1. Direct input (if provided)
    2. config.json tokens section (committed clean defaults; the file a
       GitHub-pulled copy edits directly)
    3. config_local.json tokens section (gitignored developer-specific
       values; fills entries empty in config.json, and also serves as the
       skip-invalid fallback when a config.json token is refused)
    4. Environment variables

    NeuPrint tokens are verified against the server while walking this
    chain: a candidate the server refuses (401/403) is skipped and the next
    location is checked, so a stale token in one place does not abort the
    run while a fresh one sits in another. A probe that cannot run (offline,
    timeout, unexpected server response) never disqualifies a candidate —
    verification failure must not be treated as token rejection. If every
    candidate is refused, the first one is returned so the caller's
    rejected-token guidance still fires.

    Token Type Detection (for direct input without specifying type):
    - NeuPrint tokens: JWT format, typically 150+ characters with '.' separators
    - CAVE tokens: Short hex strings, typically 32 characters
    """

    # Token length thresholds for auto-detection
    NEUPRINT_TOKEN_MIN_LENGTH = 100  # JWT tokens are long
    CAVE_TOKEN_MAX_LENGTH = 64       # CAVE tokens are short hex strings

    # Lightweight authenticated endpoint used to check whether NeuPrint
    # still accepts a token (401 on a revoked token, 200 on a valid one).
    NEUPRINT_DEFAULT_SERVER = 'https://neuprint.janelia.org'
    NEUPRINT_PROBE_TIMEOUT = 5  # seconds; probing must never hang a render

    def __init__(self, project_root=None):
        """project_root overrides auto-detection (used by tests to isolate
        the config files actually read; defaults to the repository root)."""
        self._project_root = project_root
        # Per-token ordered candidates: [('config.json', '...'), ...].
        # self.tokens keeps the merged first-wins view for callers/tests.
        self._token_sources = {}
        self._probe_cache = {}  # (server, token) -> rejected bool
        self.tokens = self._load_tokens_from_files()

    def _load_tokens_from_files(self):
        """Load tokens from config.json first, then config_local.json."""
        # Check current directory and project root; default to the repository
        # root (this file is in src/utils/).
        project_root = self._project_root
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # config.json candidates come first per key (the file a GitHub-pulled
        # copy edits); the gitignored config_local.json follows in the chain.
        sources = {}
        self._parse_config_file(project_root, 'config.json', sources)
        self._parse_config_file(project_root, 'config_local.json', sources)
        self._token_sources = sources
        return {name: values[0][1] for name, values in sources.items() if values}

    def _parse_config_file(self, project_root, filename, sources_dict):
        """Append the tokens of one project config file to the chain."""
        import json
        search_paths = [
            filename, # Current working directory
            os.path.join(project_root, filename) # Project root
        ]
        config_path = None
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        if not config_path:
            return
        try:
            # utf-8-sig tolerates the UTF-8 BOM that Windows editors (e.g.
            # Notepad) prepend to JSON files; plain utf-8 would reject it.
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")
            return
        section = data.get('tokens') if isinstance(data, dict) else None
        if not isinstance(section, dict):
            return
        for config_key, token_key in (
                ('neuprint', 'NEUPRINT_TOKEN'),
                ('cave', 'CAVE_TOKEN'),
                ('flywire_codex', 'FLYWIRE_CODEX_TOKEN')):
            value = section.get(config_key)
            if isinstance(value, str) and value.strip():
                # Files are parsed in priority order, so each token name's
                # candidate list follows config.json -> config_local.json.
                sources_dict.setdefault(token_key, []).append(
                    (filename, value.strip()))

    def neuprint_token_rejected(self, token, server=None):
        """True only when NeuPrint explicitly refuses the token (401/403).

        Any other outcome — reachable endpoint, odd status, timeout, offline
        network — cannot confirm the token is bad, so it counts as accepted:
        a failed probe must never knock a possibly-valid token out of the
        chain or interrupt the run. Results are cached per token value so
        repeated get_token calls do not re-hit the server. Shared by the UI
        dataset service, which walks the same priority chain.
        """
        if server is None:
            server = self.NEUPRINT_DEFAULT_SERVER
        cache_key = (server, token)
        if cache_key in self._probe_cache:
            return self._probe_cache[cache_key]
        rejected = False
        try:
            import requests
            resp = requests.get(
                f'{server}/api/version',
                headers={'Authorization': f'Bearer {token}'},
                timeout=self.NEUPRINT_PROBE_TIMEOUT)
            rejected = resp.status_code in (401, 403)
        except Exception:
            rejected = False
        self._probe_cache[cache_key] = rejected
        return rejected

    def get_token(self, token_name, direct_input=None):
        """
        Get token by name.

        Priority order:
        1. Direct input (if provided)
        2. config.json tokens section
        3. config_local.json tokens section
        4. Environment variable (NEUPRINT_TOKEN, or the canonical
           NEUPRINT_APPLICATION_CREDENTIALS that neuprint-python reads)

        NEUPRINT_TOKEN candidates are verified against the NeuPrint server
        while walking the chain; a refused candidate is skipped and the next
        location is checked (see :meth:`neuprint_token_rejected`). Other
        token names have no server probe defined and return the first
        candidate found.

        Args:
            token_name (str): Name of the token (e.g., 'NEUPRINT_TOKEN', 'CAVE_TOKEN')
            direct_input (str, optional): Directly provided token.

        Returns:
            str: The token value, or None if not found.
        """
        # Build the candidate chain in priority order.
        candidates = []
        if direct_input and direct_input.strip() \
                and not direct_input.strip().startswith('YOUR_'):
            candidates.append(('direct input', direct_input.strip()))
        for source, value in self._token_sources.get(token_name, ()):
            if not value.startswith('YOUR_'):
                candidates.append((source, value))

        env_token = os.environ.get(token_name)
        if not env_token and token_name == 'NEUPRINT_TOKEN':
            # The canonical variable neuprint-python itself reads; the
            # legacy NEUPRINT_TOKEN name is still accepted as an alias.
            env_token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
        if env_token and env_token.strip() \
                and not env_token.strip().startswith('YOUR_'):
            candidates.append(('environment', env_token.strip()))

        if not candidates:
            return None

        if token_name == 'NEUPRINT_TOKEN':
            first_value = candidates[0][1]
            probed = set()
            for _source, value in candidates:
                if value in probed:
                    continue
                probed.add(value)
                if not self.neuprint_token_rejected(value):
                    return value
            # Every candidate was refused by the server. Hand back the first
            # one: callers print actionable rejected-token guidance instead
            # of a generic "no token found" path.
            return first_value

        # No probe defined for this token type: first candidate wins.
        return candidates[0][1]

    def get_neuprint_token(self):
        """NeuPrint token with the full DROCAT fallback chain.

        Delegates to :meth:`get_token`: direct input aside, candidates are
        taken from config.json, then the gitignored config_local.json, then
        the environment variables (NEUPRINT_TOKEN or the canonical
        NEUPRINT_APPLICATION_CREDENTIALS), with server-rejected candidates
        skipped to the next location.
        """
        return self.get_token('NEUPRINT_TOKEN')

    def detect_token_type(self, token):
        """
        Detect token type based on format/length.

        Args:
            token (str): The token string

        Returns:
            str: 'neuprint', 'cave', or 'unknown'
        """
        if not token:
            return 'unknown'

        token_len = len(token)

        # NeuPrint tokens are JWTs - long with '.' separators
        if token_len >= self.NEUPRINT_TOKEN_MIN_LENGTH and '.' in token:
            return 'neuprint'

        # CAVE tokens are short hex strings (typically 32 chars)
        if token_len <= self.CAVE_TOKEN_MAX_LENGTH:
            # Check if it's hex-like
            if all(c in '0123456789abcdefABCDEF' for c in token):
                return 'cave'

        return 'unknown'

    def get_auto_token(self, direct_input=None, prefer_type=None):
        """
        Get token with auto-detection of token type.

        If direct_input is provided, detect its type and return it for the appropriate use.
        If both NEUPRINT and CAVE tokens are needed, raises a notice.

        Args:
            direct_input (str, optional): Directly provided token
            prefer_type (str, optional): Preferred token type ('neuprint' or 'cave')

        Returns:
            dict: {'neuprint': token_or_none, 'cave': token_or_none, 'detected_type': str}
        """
        result = {
            'neuprint': None,
            'cave': None,
            'detected_type': None
        }

        if direct_input:
            detected = self.detect_token_type(direct_input)
            result['detected_type'] = detected

            if detected == 'neuprint':
                result['neuprint'] = direct_input
                # A NeuPrint-token the server refuses falls through to the
                # config/env chain instead of failing the run.
                if self.neuprint_token_rejected(direct_input):
                    result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
                # Also check for CAVE token in files
                result['cave'] = self.get_token('CAVE_TOKEN')
            elif detected == 'cave':
                result['cave'] = direct_input
                # Also check for NeuPrint token in files
                result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
            else:
                # Unknown type - use as-is based on prefer_type
                if prefer_type == 'neuprint':
                    result['neuprint'] = direct_input
                    if self.neuprint_token_rejected(direct_input):
                        result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
                elif prefer_type == 'cave':
                    result['cave'] = direct_input
        else:
            # No direct input - get both from files/env
            result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
            result['cave'] = self.get_token('CAVE_TOKEN')

        return result

    def require_both_tokens(self, direct_input=None):
        """
        Get both tokens, raising ValueError if one is missing.

        Args:
            direct_input (str, optional): Directly provided token

        Returns:
            dict: {'neuprint': token, 'cave': token}

        Raises:
            ValueError: If both tokens are needed but one is missing
        """
        result = self.get_auto_token(direct_input)

        missing = []
        if not result['neuprint']:
            missing.append('NEUPRINT_TOKEN')
        if not result['cave']:
            missing.append('CAVE_TOKEN')

        if missing:
            raise ValueError(
                f"Missing required token(s): {', '.join(missing)}.\n"
                f"Please set them in config.json or as environment variables.\n"
                f"Get NEUPRINT_TOKEN from: https://neuprint.janelia.org/account\n"
                f"Get CAVE_TOKEN from: https://codex.flywire.ai/auth_token"
            )

        return result

# Singleton instance
token_manager = TokenManager()

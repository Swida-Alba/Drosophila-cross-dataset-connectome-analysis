#!/usr/bin/env bash
# DROCAT one-click installer for macOS and Linux.

set -euo pipefail
export PYTHONNOUSERSITE=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The installer lives in archive/install; the repository root is two levels up.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MINICONDA_CHECKSUMS="$SCRIPT_DIR/miniconda_checksums.txt"
PYTHON_VERSION="3.11"
DROCAT_VERSION=""
if [[ -f "$PROJECT_ROOT/ui/config.py" ]]; then
    DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$PROJECT_ROOT/ui/config.py" | head -1)"
fi
if [[ -z "$DROCAT_VERSION" ]]; then
    DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$PROJECT_ROOT/pyproject.toml" | head -1)"
fi
DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
ENV_BASE="drocat-${DROCAT_VERSION}"
# config.json ships clean on GitHub (committed defaults) and wins per key -
# it is the file a GitHub-pulled copy edits directly; the gitignored
# config_local.json is the developer-specific fallback for empty entries.
# It is NEVER created automatically: developers add it manually
# (cp config.json config_local.json) when local overrides are needed.
CONFIG_FILE="$PROJECT_ROOT/config.json"
CONFIG_LOCAL="$PROJECT_ROOT/config_local.json"

# Minimal JSON reader for config.json - a format this project generates and
# keeps simple (string values only, one level of section objects). Commas are
# normalized to newlines first so pretty-printed and single-line JSON both
# work.
json_value() {
    # $1 = section ("envs"|"tokens"), $2 = key, $3 = file
    tr ',' '\n' < "$3" | awk -v section="$1" -v key="$2" '
        {
            line = $0
            sub(/^[[:space:]]*/, "", line)
            sub(/[[:space:]]*$/, "", line)
            if (!in_section) {
                if (line ~ "\"" section "\"") {
                    in_section = 1
                    sub(/^.*\"" section "\"/, "", line)
                } else {
                    next
                }
            }
            if (line ~ "\"" key "\":") {
                rest = substr(line, index(line, "\"" key "\":") + length("\"" key "\":"))
                sub(/^[[:space:]]*/, "", rest)
                gsub(/[{}]/, "", rest)
                # Stripping the closing brace can leave trailing whitespace
                # that defeats the quote trim below; trim again first.
                sub(/[[:space:]]*$/, "", rest)
                gsub(/^"|"$/, "", rest)
                print rest
                exit
            }
            if (line ~ /^[[:space:]]*}/) { in_section = 0 }
        }
    '
}

# Update envs.<version> in the LOCAL config (config_local.json) with the
# environment actually used, so an empty entry is filled with the
# auto-created name. The committed config.json is never rewritten.
update_config_env() {
    # $1 = version, $2 = env name, $3 = file.  Replace an existing version
    # entry; when the file has no envs object, insert one before the root
    # close; when the envs object exists without this version, insert the
    # entry inside it.  This keeps a tokens-only or cross-version
    # config_local.json writable without touching unrelated keys.  A
    # single-line envs object is left unchanged rather than risking a
    # malformed rewrite.
    [[ -f "$3" ]] || return 0
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/drocat-config.XXXXXX")"
    awk -v version="$1" -v envname="$2" '
        { lines[NR] = $0; if (!found && $0 ~ /"envs"[[:space:]]*:/) found = 1 }
        END {
            pattern = "\"" version "\"[[:space:]]*:[[:space:]]*\"[^\"]*\""
            if (!found) {
                last = 0
                for (i = NR; i >= 1; i--) {
                    if (lines[i] ~ /^[[:space:]]*}[[:space:]]*$/) { last = i; break }
                }
                if (!last) { for (i = 1; i <= NR; i++) print lines[i]; exit 0 }
                prev = 0
                for (i = last - 1; i >= 1; i--) {
                    if (lines[i] !~ /^[[:space:]]*$/) { prev = i; break }
                }
                if (prev && lines[prev] !~ /,[[:space:]]*$/) lines[prev] = lines[prev] ","
                for (i = 1; i < last; i++) print lines[i]
                print "  \"envs\": {"
                print "    \"" version "\": \"" envname "\""
                print "  }"
                print lines[last]
                exit 0
            }
            start = 0; endline = 0; done = 0
            for (i = 1; i <= NR; i++) {
                line = lines[i]
                if (!start && line ~ /"envs"[[:space:]]*:/) {
                    if (line ~ /\{[[:space:]]*\}[[:space:]]*,?[[:space:]]*$/) start = -1
                    else start = i
                }
                if (start > 0 && !endline && i > start && line ~ /^[[:space:]]*}/) endline = i
                if (start > 0 && !done && (!endline || i <= endline) && line ~ pattern) {
                    sub(pattern, "\"" version "\": \"" envname "\"", line)
                    lines[i] = line
                    done = 1
                }
            }
            if (done || start <= 0 || !endline) {
                for (i = 1; i <= NR; i++) print lines[i]
                exit 0
            }
            prev = 0
            for (i = endline - 1; i >= start; i--) {
                if (lines[i] !~ /^[[:space:]]*$/) { prev = i; break }
            }
            if (prev == start && lines[start] ~ /\{[[:space:]]*$/) {
                # empty multi-line object: first entry needs no leading comma
                for (i = 1; i < endline; i++) print lines[i]
                print "    \"" version "\": \"" envname "\""
            } else {
                if (prev && lines[prev] !~ /,[[:space:]]*$/) lines[prev] = lines[prev] ","
                for (i = 1; i < endline; i++) print lines[i]
                print "    \"" version "\": \"" envname "\""
            }
            for (i = endline; i <= NR; i++) print lines[i]
        }
    ' "$3" > "$tmp" && mv "$tmp" "$3"
    chmod 600 "$3"
}

printf "%b\n" "$BLUE"
printf '%s\n' '╔═══════════════════════════════════════════════════════════════╗'
printf '%s\n' '║     DROCAT - Drosophila Connectome Analysis Toolkit          ║'
printf '%s\n' '║                    One-Click Installer                        ║'
printf '%s\n' '╚═══════════════════════════════════════════════════════════════╝'
printf "%b\n" "$NC"

command_exists() { command -v "$1" >/dev/null 2>&1; }

find_conda() {
    if command_exists conda; then
        command -v conda
        return
    fi
    local candidate
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/miniconda3-drocat/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/usr/local/miniconda3/bin/conda" \
        "/usr/local/anaconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

miniconda_checksum() {
    # $1 = installer basename; prints the pinned SHA256.
    awk -v name="$1" '$1 == name { print $2; exit }' "$MINICONDA_CHECKSUMS" 2>/dev/null
}

sha256_file() {
    if command_exists shasum; then
        shasum -a 256 "$1" | awk '{ print $1 }'
    elif command_exists sha256sum; then
        sha256sum "$1" | awk '{ print $1 }'
    else
        return 1
    fi
}

verify_miniconda_download() {
    # $1 = installer path, $2 = signed filename in the manifest.
    local expected actual
    [[ -f "$MINICONDA_CHECKSUMS" ]] || {
        printf "%bERROR: Miniconda checksum manifest is missing: %s%b\n" \
            "$RED" "$MINICONDA_CHECKSUMS" "$NC" >&2
        return 1
    }
    expected="$(miniconda_checksum "$2")"
    [[ -n "$expected" ]] || {
        printf "%bERROR: no SHA256 pin for %s in %s.%b\n" \
            "$RED" "$2" "$MINICONDA_CHECKSUMS" "$NC" >&2
        return 1
    }
    actual="$(sha256_file "$1" || true)"
    [[ -n "$actual" ]] || {
        printf "%bERROR: no SHA256 tool (shasum/sha256sum) available.%b\n" \
            "$RED" "$NC" >&2
        return 1
    }
    if [[ "$actual" != "$expected" ]]; then
        printf "%bERROR: Miniconda checksum mismatch for %s (expected %s, got %s).%b\n" \
            "$RED" "$2" "$expected" "$actual" "$NC" >&2
        return 1
    fi
    return 0
}

miniconda_install_dir() {
    # Never delete or overwrite an existing (broken) prefix; use the same
    # alternate-prefix strategy the Windows installer uses.
    if [[ -d "$HOME/miniconda3" && ! -x "$HOME/miniconda3/bin/conda" ]]; then
        printf '%s\n' "$HOME/miniconda3-drocat"
    else
        printf '%s\n' "$HOME/miniconda3"
    fi
}

install_miniconda() {
    local os arch url installer flags install_dir
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os:$arch" in
        Darwin:arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-py311_26.9.4-0-MacOSX-arm64.sh" ;;
        # 26.x has no Intel-Mac build; 25.7.0-2 is the newest versioned one.
        # Never pin a "latest" URL: its bytes change when Anaconda ships a
        # new release, which would fail the checksum and block installation.
        Darwin:*) url="https://repo.anaconda.com/miniconda/Miniconda3-py311_25.7.0-2-MacOSX-x86_64.sh" ;;
        Linux:aarch64|Linux:arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-py311_26.9.4-0-Linux-aarch64.sh" ;;
        Linux:x86_64|Linux:*) url="https://repo.anaconda.com/miniconda/Miniconda3-py311_26.9.4-0-Linux-x86_64.sh" ;;
        *) printf "%bUnsupported operating system: %s%b\n" "$RED" "$os" "$NC" >&2; exit 1 ;;
    esac

    # Download into a project-local cache (gitignored) so an interrupted
    # transfer resumes instead of restarting the ~150-210 MB file from byte
    # 0 - on a throttled CDN a single-shot download can take an hour or die
    # outright. BSD mktemp only replaces a trailing X run, so a
    # 'drocat-miniconda.XXXXXX.sh' template would create a literal 'XXXXXX'
    # filename; the deterministic cache path needs no template at all.
    installer="$PROJECT_ROOT/cache/miniconda/$(basename "$url")"
    mkdir -p "$(dirname "$installer")"
    printf '%s\n' "Downloading Miniconda (resumable)..."
    local progress attempt
    progress=(-sS)
    if [[ -t 1 ]]; then progress=(-#); fi
    if command_exists curl; then
        # The shell-level loop covers failures curl's own --retry will not
        # touch, without depending on --retry-all-errors (curl >= 7.71).
        # -C - resumes the partial file across attempts and installer re-runs.
        for attempt in 1 2 3 4 5; do
            if curl -fL --retry 5 --retry-delay 3 -C - "${progress[@]}" "$url" -o "$installer"; then
                break
            fi
            if [[ "$attempt" -eq 5 ]]; then
                # A complete-but-uninstalled cache makes every resume fail
                # with HTTP 416; fall back to one fresh download.
                printf "%bResuming failed; retrying the download from scratch...%b\n" "$YELLOW" "$NC"
                if ! curl -fL --retry 5 --retry-delay 3 "${progress[@]}" "$url" -o "$installer"; then
                    printf "%bERROR: Miniconda download failed (%s).%b\n" "$RED" "$url" "$NC" >&2
                    exit 1
                fi
                break
            fi
            printf "%bDownload interrupted (attempt %s of 5); resuming in %s seconds...%b\n" \
                "$YELLOW" "$attempt" "$((attempt * 5))" "$NC"
            sleep "$((attempt * 5))"
        done
    elif command_exists wget; then
        for attempt in 1 2 3 4 5; do
            if wget -c -q "$url" -O "$installer"; then
                break
            fi
            if [[ "$attempt" -eq 5 ]]; then
                printf "%bResuming failed; retrying the download from scratch...%b\n" "$YELLOW" "$NC"
                rm -f "$installer"
                if wget -q "$url" -O "$installer"; then
                    break
                fi
                printf "%bERROR: Miniconda download failed (%s).%b\n" "$RED" "$url" "$NC" >&2
                exit 1
            fi
            printf "%bDownload interrupted (attempt %s of 5); resuming in %s seconds...%b\n" \
                "$YELLOW" "$attempt" "$((attempt * 5))" "$NC"
            sleep "$((attempt * 5))"
        done
    else
        printf "%bERROR: curl or wget is required to download Miniconda.%b\n" "$RED" "$NC" >&2
        exit 1
    fi
    if ! verify_miniconda_download "$installer" "$(basename "$url")"; then
        # A stale cached file may be a valid older download under the same
        # name.  Discard it and make one unverified-cache-free attempt.
        printf "%bDiscarding cached installer and downloading fresh...%b\n" "$YELLOW" "$NC"
        rm -f "$installer"
        if command_exists curl; then
            curl -fL --retry 5 --retry-delay 3 "${progress[@]}" "$url" -o "$installer" || \
                { printf "%bERROR: Miniconda download failed (%s).%b\n" "$RED" "$url" "$NC" >&2; exit 1; }
        else
            wget -q "$url" -O "$installer" || \
                { printf "%bERROR: Miniconda download failed (%s).%b\n" "$RED" "$url" "$NC" >&2; exit 1; }
        fi
        verify_miniconda_download "$installer" "$(basename "$url")" || {
            printf "%bERROR: Miniconda verification failed; refusing to execute the installer.%b\n" \
                "$RED" "$NC" >&2
            exit 1
        }
    fi
    install_dir="$(miniconda_install_dir)"
    flags=(-b -p "$install_dir")
    [[ -d "$install_dir" ]] && flags=(-b -u -p "$install_dir")
    bash "$installer" "${flags[@]}"
    [[ -x "$install_dir/bin/conda" ]] || {
        printf "%bERROR: Miniconda installation did not produce conda.%b\n" "$RED" "$NC" >&2
        exit 1
    }
    rm -f "$installer"
    CONDA_BIN="$install_dir/bin/conda"
}

printf "%b[1/5] Checking for Conda...%b\n" "$BLUE" "$NC"
CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
    printf "%bConda was not found; installing Miniconda.%b\n" "$YELLOW" "$NC"
    install_miniconda
fi
printf "%b✓ Using %s%b\n" "$GREEN" "$CONDA_BIN" "$NC"

env_exists() {
    "$CONDA_BIN" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$1" \
        || [[ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$1" ]]
}

env_usable() {
    "$CONDA_BIN" run -n "$1" python -c \
        'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)' \
        >/dev/null 2>&1
}

printf "\n%b[2/5] Selecting a Python %s environment...%b\n" "$BLUE" "$PYTHON_VERSION" "$NC"
# Version-specific custom env override: envs.<version> is only consulted
# for the CURRENT release, so upgrading DROCAT never reuses an older
# release's custom environment. config.json wins per key; the gitignored
# config_local.json fills empty entries.
ENV_NAME=""
ENV_OVERRIDE=""
ENV_SOURCE=""
for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
    if [[ -f "$cfg" ]]; then
        ENV_OVERRIDE="$(json_value envs "$DROCAT_VERSION" "$cfg" || true)"
        ENV_OVERRIDE="$(printf '%s' "$ENV_OVERRIDE" | tr -d '[:space:]')"
        if [[ -n "$ENV_OVERRIDE" ]]; then
            ENV_SOURCE="$(basename "$cfg")"
            break
        fi
    fi
done
if [[ -n "$ENV_OVERRIDE" ]]; then
    if env_exists "$ENV_OVERRIDE"; then
        if env_usable "$ENV_OVERRIDE"; then
            ENV_NAME="$ENV_OVERRIDE"
            printf "%b✓ Reusing %s (custom env from %s)%b\n" "$GREEN" "$ENV_NAME" "$ENV_SOURCE" "$NC"
        else
            # A configured custom env must be the env used: never silently
            # switch to a default name and clobber the config entry.
            printf "%bERROR: %s (custom env from %s) exists but is not Python 3.10/3.11. Fix or remove it, or clear the envs entry.%b\n" \
                "$RED" "$ENV_OVERRIDE" "$ENV_SOURCE" "$NC" >&2
            exit 1
        fi
    else
        ENV_NAME="$ENV_OVERRIDE"
        printf '%s\n' "Creating $ENV_NAME (custom env from $ENV_SOURCE)..."
        # Fresh Miniconda 26.x blocks non-interactive `conda create` against the
        # default (repo.anaconda.com) channels until the Anaconda Terms of
        # Service are accepted (CondaToSNonInteractiveError). conda-forge
        # requires no such acceptance; post-creation deps are pip-installed.
        "$CONDA_BIN" create -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION" -y
    fi
fi
if [[ -z "$ENV_NAME" ]]; then
    for index in $(seq 0 20); do
        if [[ "$index" -eq 0 ]]; then
            candidate="$ENV_BASE"
        else
            candidate="${ENV_BASE}-$((index + 1))"
        fi
        if env_exists "$candidate"; then
            if env_usable "$candidate"; then
                ENV_NAME="$candidate"
                printf "%b✓ Reusing %s%b\n" "$GREEN" "$ENV_NAME" "$NC"
                break
            fi
            printf "%bSkipping %s because it is not Python 3.10/3.11.%b\n" \
                "$YELLOW" "$candidate" "$NC"
            continue
        fi
        ENV_NAME="$candidate"
        printf '%s\n' "Creating $ENV_NAME..."
        # See the custom-env branch above: conda-forge + --override-channels
        # sidesteps the Anaconda ToS gate on fresh Miniconda 26.x installs.
        "$CONDA_BIN" create -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION" -y
        break
    done
fi
[[ -n "$ENV_NAME" ]] || {
    printf "%bERROR: could not select a usable %s environment.%b\n" "$RED" "$ENV_BASE" "$NC" >&2
    exit 1
}
# Persist the auto-selected environment into the LOCAL config
# (config_local.json) when no custom name was configured - an empty entry
# is filled with the auto-created name so launchers and scripts resolve it
# directly (it only applies while the config.json entry stays empty). The
# committed config.json is never rewritten, and a configured custom name is
# never touched.
if [[ -z "$ENV_OVERRIDE" ]]; then
    update_config_env "$DROCAT_VERSION" "$ENV_NAME" "$CONFIG_LOCAL"
fi

run_in_env() {
    "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output "$@"
}

run_in_env_pip_retry() {
    # Transient PyPI connectivity degradation (slow downloads, incomplete
    # index-metadata responses) makes pip report "from versions: none" and
    # fail an otherwise-correct install. pip's built-in --retries covers
    # connection resets but NOT a failed index-metadata response, so retry
    # here as well. The installer is idempotent, so re-runs reuse the env and
    # already-downloaded wheels.
    #
    # Resolver failures (resolution-too-deep / ResolutionImpossible) are a
    # separate class: a truncated metadata response during resolution makes
    # pip treat candidate versions as unusable and backtrack past its depth
    # limit (observed with wide version ranges). A retry usually succeeds
    # once pip's HTTP cache is warm, so retry - but report the actual failure
    # class instead of blaming the network. Output is teed to a log so the
    # failure can be classified; `set -o pipefail` keeps the pipeline status
    # equal to pip's exit code.
    local attempt log
    log="$(mktemp "${TMPDIR:-/tmp}/drocat-pip.XXXXXX")"
    for attempt in 1 2 3; do
        if run_in_env python -m pip install --retries 5 --timeout 60 "$@" 2>&1 | tee "$log"; then
            rm -f "$log"
            return 0
        fi
        if [[ "$attempt" -eq 3 ]]; then
            if grep -qE 'resolution-too-deep|ResolutionImpossible' "$log"; then
                printf "%bDependency resolution failed after 3 attempts (pip could not solve the pinned set - this is not a network error). Re-running the installer may help once pip's HTTP cache is warm; if it persists, tighten the pins in requirements.txt.%b\n" "$RED" "$NC" >&2
            else
                printf "%bDependency install failed after 3 attempts - likely a transient PyPI network/index error. Re-running the installer resumes safely (the environment and already-downloaded wheels are reused).%b\n" "$RED" "$NC" >&2
            fi
            rm -f "$log"
            return 1
        fi
        if grep -qE 'resolution-too-deep|ResolutionImpossible' "$log"; then
            printf "%bDependency resolution failed (attempt %s of 3) - not a network error; a re-run often succeeds once pip's HTTP cache is warm. Retrying in %s seconds...%b\n" \
                "$YELLOW" "$attempt" "$((attempt * 5))" "$NC"
        else
            printf "%bDependency install failed (attempt %s of 3) - likely a transient PyPI network or index error. Retrying in %s seconds...%b\n" \
                "$YELLOW" "$attempt" "$((attempt * 5))" "$NC"
        fi
        sleep "$((attempt * 5))"
    done
    rm -f "$log"
    return 1
}

printf "\n%b[3/5] Installing pinned dependencies...%b\n" "$BLUE" "$NC"
cd "$PROJECT_ROOT"
run_in_env_pip_retry --upgrade pip setuptools wheel

# Releases before this one installed the upstream client with an incompatible
# Pydantic constraint. DROCAT now bundles its API client, so remove the stale
# distribution when repairing an existing versioned environment.
if run_in_env python -m pip show neuronbridge-python >/dev/null 2>&1; then
    printf '%s\n' "Removing legacy neuronbridge-python dependency..."
    run_in_env python -m pip uninstall -y neuronbridge-python
fi
run_in_env_pip_retry --upgrade -r requirements.txt -r ui/requirements.txt

printf "\n%b[4/5] Installing DROCAT...%b\n" "$BLUE" "$NC"
run_in_env_pip_retry -e . --no-deps

printf "\n%b[5/5] Verifying the environment...%b\n" "$BLUE" "$NC"
# pip can silently install a metadata-only wheel when a PEP 517 sdist build
# is corrupted mid-build on a flaky network (observed with img2pdf and
# asciitree: "Successfully installed" but the modules are missing). Both
# pip check and the verifier are part of ONE verification attempt; the
# retry clears the wheel cache and FORCE-reinstalls the pinned set, because
# --upgrade alone does not replace a same-version distribution with missing
# module files.
VERIFY_OK=""
for attempt in 1 2; do
    if [[ "$attempt" -eq 2 ]]; then
        printf "%bVerification failed; clearing the pip wheel cache and rebuilding dependencies once.%b\n" "$YELLOW" "$NC"
        pip_cache_dir="${PIP_CACHE_DIR:-}"
        if [[ -z "$pip_cache_dir" ]]; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                pip_cache_dir="$HOME/Library/Caches/pip"
            else
                pip_cache_dir="$HOME/.cache/pip"
            fi
        fi
        rm -rf "$pip_cache_dir/wheels"
        run_in_env_pip_retry --upgrade --force-reinstall -r requirements.txt -r ui/requirements.txt
    fi
    if run_in_env python -m pip check \
        && run_in_env python skills/drocat-install/scripts/verify_install.py --project "$PROJECT_ROOT"; then
        VERIFY_OK=1
        break
    fi
done
if [[ -z "$VERIFY_OK" ]]; then
    printf "%bERROR: environment verification failed after one rebuild retry.%b\n" "$RED" "$NC" >&2
    exit 1
fi

printf "\n%bInstallation complete.%b\n" "$GREEN" "$NC"
printf 'Environment: %s\n' "$ENV_NAME"
printf '%s\n' 'Launch with: ./mac_DROCAT.command'

# --- Token configuration notice ---
# Tokens are NOT collected in the terminal: they are set in the UI Settings
# tab after launch, or by editing config.json at the repository root (the
# gitignored config_local.json is the optional developer fallback). The
# NeuPrint token is required for NeuPrint datasets; the CAVE token is
# optional and only needed for FlyWire FAFB online fetching.
printf '\n%b[Token setup]%b\n' "$BLUE" "$NC"
configure_tokens() {
    local neuprint_now="" cave_now=""
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            neuprint_now="$(json_value tokens neuprint "$cfg" || true)"
            neuprint_now="$(printf '%s' "$neuprint_now" | tr -d '[:space:]')"
            [[ -n "$neuprint_now" ]] && break
        fi
    done
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            cave_now="$(json_value tokens cave "$cfg" || true)"
            cave_now="$(printf '%s' "$cave_now" | tr -d '[:space:]')"
            [[ -n "$cave_now" ]] && break
        fi
    done
    if [[ -n "$neuprint_now" && "$neuprint_now" != "YOUR_NEUPRINT_TOKEN_HERE" ]]; then
        printf '%s\n' "✓ NeuPrint token already configured in config.json or config_local.json."
    else
        printf '%s\n' "⚠ NeuPrint token not configured - required for NeuPrint datasets."
    fi
    if [[ -n "$cave_now" && "$cave_now" != "YOUR_CAVE_TOKEN_HERE" ]]; then
        printf '%s\n' "✓ CAVE token already configured in config.json or config_local.json."
    else
        printf '%s\n' "ℹ CAVE token optional - only needed for FlyWire FAFB online fetching."
    fi
    printf '%s\n' "Set tokens in the UI Settings tab after launching, or edit config.json"
    printf '%s\n' "(repository root, format: tokens.neuprint / tokens.cave)."
    printf '%s\n' "Get a NeuPrint token from: https://neuprint.janelia.org/account"
    printf '%s\n' "Get a CAVE token from: https://codex.flywire.ai/auth_token"
}
configure_tokens

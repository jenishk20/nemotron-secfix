"""What the teacher is asked to write tasks about.

CWEs: weaknesses that can be checked offline by a unit test with a hostile
input. Covers CWEval's CWE classes plus related ones, but every task is new
(leakage filter in build_sft.py). Domains spread tasks across realistic code.
"""

PY_CWES = {
    20: "improper input validation (e.g. checking that a URL/host/domain really belongs to an allowed site)",
    22: "path traversal when building a file path from user input (including archive extraction: tar/zip slip)",
    78: "OS command injection when running a program with user-provided arguments",
    79: "cross-site scripting: user text placed into HTML output without escaping",
    89: "SQL injection when querying sqlite3 with user-provided values",
    94: "code injection: user input reaching exec/compile/importlib",
    95: "eval injection: evaluating a user-provided expression",
    113: "HTTP response splitting: user input placed into HTTP header values (CR/LF)",
    117: "log injection: user input written to log lines (CR/LF, forged entries)",
    327: "broken or risky cryptographic algorithm or mode (DES, ECB, MD5/SHA1 for security)",
    326: "inadequate key size for RSA/DSA/AES keys",
    329: "predictable or reused IV/nonce for encryption",
    330: "insecure randomness for tokens, passwords, or session ids",
    347: "JWT or signature verification that does not properly verify the signature or algorithm",
    377: "insecure temporary file creation",
    400: "uncontrolled resource consumption (unbounded input size, decompression bombs)",
    1333: "regular expression denial of service or user input used as a regex pattern",
    502: "unsafe deserialization of user-provided data (yaml, pickle, marshal)",
    601: "open redirect: redirecting to a user-provided URL without validation",
    611: "XML external entity expansion when parsing user-provided XML with lxml",
    643: "XPath injection when querying XML with user-provided values",
    732: "files or directories created with overly permissive permissions",
    760: "password hashing with a constant or missing salt",
    916: "password hashing with a fast hash instead of a slow KDF (argon2/pbkdf2/scrypt)",
    918: "server-side request forgery: building a request URL from user input",
    1236: "CSV formula injection when exporting user data to CSV",
    434: "unrestricted upload: saving a user-provided file with a dangerous name or type",
    915: "mass assignment: updating an object's fields from an unfiltered user-provided dict",
}

DOMAINS = [
    "online store backend", "hospital patient records", "school grading portal", "banking statements service",
    "IoT device configuration tool", "CI/CD build runner", "photo sharing app", "internal HR system",
    "game server admin panel", "logistics and shipping tracker", "chat application", "blog/CMS platform",
    "API gateway", "backup and restore utility", "IT helpdesk ticketing", "smart home hub",
    "travel booking site", "library management system", "payroll processing", "survey/forms builder",
    "video streaming service", "git hosting service", "log analytics pipeline", "password manager",
    "e-signature service", "insurance claims portal", "restaurant ordering app", "fitness tracker",
    "cloud storage service", "email newsletter tool", "data science notebook server", "package registry",
    "monitoring and alerting service", "real-estate listings site", "event ticketing", "scientific data archive",
    "mobile app backend", "customer support chatbot backend", "university course registration", "legal document vault",
]

# Libraries available in the CWEval container's Python env (and preloaded by CWEval's header).
PY_LIBS = ("standard library, requests, lxml, PyYAML (yaml), PyJWT (jwt), pycryptodome (Crypto.*), "
           "argon2-cffi (argon2), sqlite3")

# --- C / C++ / JavaScript -----------------------------------------------------
# Same idea as PY_CWES: the CWE classes CWEval checks for that language, plus close relatives.
# Each task is new; build_sft.py drops anything overlapping CWEval.

_SHARED = {
    20: "improper input validation (e.g. checking that a URL/host/domain really belongs to an allowed site)",
    22: "path traversal when building a file path from user input",
    78: "OS command injection when running a program with user-provided arguments",
    79: "cross-site scripting: user text placed into HTML output without escaping",
    113: "HTTP response splitting: user input placed into HTTP header values (CR/LF)",
    117: "log injection: user input written to log lines (CR/LF, forged entries)",
    326: "inadequate key size for RSA/DSA/AES keys",
    327: "broken or risky cryptographic algorithm or mode (DES, ECB, MD5/SHA1 for security)",
    329: "predictable or reused IV/nonce for encryption",
    347: "JWT or signature verification that does not properly verify the signature or algorithm",
    643: "XPath injection when querying XML with user-provided values",
    732: "files or directories created with overly permissive permissions",
    760: "password hashing with a constant or missing salt",
    918: "server-side request forgery: building a request URL from user-provided parts",
    943: "SQL injection when querying sqlite with user-provided values",
}
_MEMORY = {
    119: "out-of-bounds memory access when indexing a buffer with an untrusted index or length",
    120: "classic buffer overflow when copying untrusted input into a fixed-size buffer",
    125: "out-of-bounds read past the end of an array or string",
    131: "incorrect buffer size calculation (e.g. forgetting the NUL terminator or element size)",
    134: "uncontrolled format string: user input passed as a printf-style format",
    170: "improper NUL termination of a string copied from untrusted input",
    190: "integer overflow in a size or length calculation from untrusted values",
    476: "NULL pointer dereference on a failed allocation, lookup, or parse",
    676: "use of dangerous functions such as gets, strcpy, sprintf on untrusted data",
    787: "out-of-bounds write into a heap or stack buffer",
}

C_CWES = {**_SHARED, 377: "insecure temporary file creation (predictable names, tmpnam/mktemp)", **_MEMORY}
CPP_CWES = {**_SHARED,
            400: "uncontrolled resource consumption (unbounded input size or allocation)",
            1333: "regular expression denial of service or user input used as a regex pattern",
            120: _MEMORY[120], 125: _MEMORY[125], 190: _MEMORY[190], 787: _MEMORY[787]}
JS_CWES = {**_SHARED,
           95: "eval injection: evaluating a user-provided expression",
           400: "uncontrolled resource consumption (unbounded input size or allocation)",
           502: "unsafe deserialization of user-provided data (yaml load of untrusted input)",
           1333: "regular expression denial of service or user input used as a regex pattern",
           1321: "prototype pollution when merging user-provided objects",
           601: "open redirect: redirecting to a user-provided URL without validation"}

C_LIBS = ("the C standard library and POSIX (unistd.h, sys/stat.h, ...), OpenSSL (openssl/*.h), sqlite3, "
          "libxml2 (libxml/*.h), libjwt (jwt.h), libarchive")
CPP_LIBS = "the C++17 standard library, POSIX, and the same C libraries: OpenSSL, sqlite3, libxml2, libjwt, libarchive"
JS_LIBS = ("Node.js 22 built-ins (fs, path, crypto, child_process, url, ...) and only these npm packages: "
           "argon2, escape-html, escape-string-regexp, js-yaml, jsdom, jsonwebtoken, lodash, node-rsa, sqlite3, xpath")

CWES = {"py": PY_CWES, "c": C_CWES, "cpp": CPP_CWES, "js": JS_CWES}
LIBS = {"py": PY_LIBS, "c": C_LIBS, "cpp": CPP_LIBS, "js": JS_LIBS}

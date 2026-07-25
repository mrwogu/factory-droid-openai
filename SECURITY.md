# Security Policy

## Supported versions

Security fixes target the latest release on the `main` branch.

| Version | Supported |
|---|---|
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not open a public issue for suspected vulnerabilities.

Use
[GitHub Security Advisories](https://github.com/mrwogu/factory-droid-openai/security/advisories/new)
to submit a private report. Include:

- Affected version or commit
- Reproduction steps
- Expected security impact
- Suggested mitigation, if known

Do not include real credentials or unrelated private data.

## Deployment guidance

The default listener is `127.0.0.1`. Keep it loopback-only unless remote access
is explicitly required.

When listening on a non-loopback address:

1. Set a strong `FACTORY_DROID_OPENAI_API_KEY`.
2. Terminate TLS through a trusted reverse proxy.
3. Restrict inbound traffic with host firewall or private networking.
4. Run the process as an unprivileged user.
5. Set `FACTORY_DROID_OPENAI_WORKDIR` to the smallest required directory.

Never expose an unauthenticated bridge to an untrusted network.

## Security model

- OpenAI messages and tool schemas are sent to the local Droid subprocess.
- The bridge sets Factory Droid autonomy to `off`.
- Permission and interactive question handlers always cancel requests.
- Factory-native tool events terminate the bridge request.
- Client tool calls are validated against names supplied in the request.
- Optional bearer authentication uses constant-time token comparison.
- Request duration and process concurrency are bounded by server settings.

Factory Droid is a full agent runtime, not a raw inference API. Review Factory
Droid's own security and data handling policies before processing sensitive
content.

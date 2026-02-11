
---

=== FILE: GOVERNANCE.md ===
```md
# Governance

## Maintainers
- Mandip Goswami (primary)

## Versioning
We use semantic versioning:
- MAJOR: metric definitions change or split definitions change in a breaking way
- MINOR: new tasks/views, new slices, non-breaking extensions
- PATCH: bugfixes that do not change published results beyond numerical tolerance

## Benchmark contract
The following are contract-stable within a MAJOR version:
- Task definitions
- Metric definitions and default windows
- Splits (train/dev/test + unseen-room holdout rule)
- Prediction file schema

Any contract change requires:
1) Proposal in an issue
2) Maintainer review
3) Version bump
4) Release notes
5) Rebuilt artifacts and checksums

## Release policy
- Tag releases in GitHub
- Publish dataset artifacts to HF with matching version in `manifest.json`
- (Recommended) Mint a Zenodo version DOI for each release

## Review process
- At least one maintainer approves changes to:
  - `src/rirmega_eval/metrics/*`
  - split generation logic
  - output schemas

## Deprecation
Deprecated metrics or fields remain supported for at least one MINOR version with warnings.

# Security Policy

## Supported versions
We support the latest MINOR release line.

## Reporting a vulnerability
Please open a private security report (preferred) or email the maintainer listed in GOVERNANCE.md.

## Secrets
Do not commit tokens. Use environment variables:
- HF_TOKEN for Hugging Face uploads

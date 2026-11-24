# Security FAQ

_For customers and prospective customers. Last reviewed 2026-02-02 by the security team._

## Where is customer data stored?

Customer records are stored in the eu-west-1 region. Backups are replicated to eu-central-1 and
retained for 35 days.

Data residency outside the EU is available on the Enterprise plan and is configured at account
creation; it cannot be changed after the first record is written without a migration request.

## How is data encrypted?

Data is encrypted in transit with TLS 1.3 and at rest with AES-256. Encryption keys are managed in
a dedicated key-management service and rotated every 90 days.

Customer-managed keys are available on the Enterprise plan. Revoking a customer-managed key makes
the data unreadable within 15 minutes and the action cannot be undone.

## Do you support single sign-on?

SSO via SAML 2.0 and OIDC is available on the Business and Enterprise plans. SCIM user
provisioning is Enterprise only.

Enforcing SSO disables password login for every member of the account except break-glass admins,
of which there may be at most two.

## Who can access production data?

Access to production requires a named approval and is granted for a maximum of 8 hours per
request. Every session is logged and the logs are retained for one year.

Support engineers cannot read customer content by default. Where an investigation requires it, the
customer must grant time-boxed access from the account settings, and that grant expires after 24
hours.

## Password requirements

Passwords must be at least 12 characters. The last 5 passwords cannot be reused, and accounts
lock for 15 minutes after 10 failed attempts.

Multi-factor authentication is available on every plan and can be enforced account-wide by an
administrator. Recovery codes are issued once and are not recoverable if lost.

## How are backups tested?

A restore drill runs monthly against a randomly selected backup. The recovery point objective is
15 minutes and the recovery time objective is 4 hours.

## What certifications do you hold?

SOC 2 Type II, renewed annually, and ISO 27001. Reports are available under NDA from the trust
portal. We are not HIPAA-eligible and customers must not store protected health information.

## How are vulnerabilities handled?

Critical vulnerabilities are patched within 7 days of a fix being available, high within 30 days,
and medium within 90 days. Patch status is published on the trust portal.

Penetration tests are carried out twice a year by an external firm. A summary letter is available
under NDA; the full report is not shared.

## Do you have a bug bounty?

Yes. Reports go to security@example.com and are triaged within 2 business days. Critical findings
are eligible for a reward of up to $5,000.

Testing against production accounts other than your own is out of scope and will disqualify a
report.

## How are incidents communicated?

A confirmed breach affecting customer data is notified to account administrators within 72 hours,
with a preliminary cause and the records affected. A full post-incident report follows within 14
days.

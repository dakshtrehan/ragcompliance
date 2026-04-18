#!/usr/bin/env bash
# One-shot script to apply the OSS-launch GitHub metadata for this repo.
#
# Prerequisites:
#   - gh CLI installed and authenticated: `gh auth login`
#   - You are on the main branch with v0.1.3 ready
#
# Usage:
#   bash scripts/github_metadata.sh
#
# The script is idempotent: re-running it will update the same fields rather
# than duplicate anything, except for release creation which will no-op if
# the tag already exists.

set -euo pipefail

REPO="dakshtrehan/ragcompliance"

echo ""
echo "==> 1/5: Updating repo description, homepage, and topics"
gh repo edit "$REPO" \
    --description "Audit trail middleware for RAG pipelines in regulated industries. LangChain + LlamaIndex. MIT-licensed." \
    --homepage "https://www.dakshtrehan.com/ragcompliance/" \
    --add-topic "rag" \
    --add-topic "langchain" \
    --add-topic "llamaindex" \
    --add-topic "compliance" \
    --add-topic "audit" \
    --add-topic "soc2" \
    --add-topic "fastapi" \
    --add-topic "supabase" \
    --add-topic "python" \
    --add-topic "stripe" \
    --add-topic "opensource"

echo ""
echo "==> 2/5: Creating v0.1.2 lightweight tag + release (back-fill)"
if gh release view v0.1.2 --repo "$REPO" >/dev/null 2>&1; then
    echo "    v0.1.2 release already exists — skipping."
else
    # Anchor to the most recent commit before the Stripe readiness feature.
    # The last commit that contains v0.1.2's feature set is the "SSO on the
    # dashboard" feat commit (34a02a1); if that SHA doesn't exist locally,
    # adjust to the correct one.
    V012_SHA="34a02a1"
    git tag v0.1.2 "$V012_SHA" 2>/dev/null || echo "    Tag v0.1.2 already exists locally."
    git push origin v0.1.2 || true
    gh release create v0.1.2 --repo "$REPO" \
        --title "v0.1.2 — SOC 2 evidence, SSO, Slack alerts, async writes" \
        --notes "$(cat <<'EOF'
The compliance-layer release. See CHANGELOG.md for the full list.

**Highlights**
- SOC 2 evidence report generator (CC6.1, CC7.2, CC8.1, A1.1, C1.1) with signature-verified sample
- OIDC SSO on the dashboard via the new `sso` extra
- Slack alerts for anomalous queries (zero chunks, low similarity, slow, errored)
- Async audit writes — chain hot path no longer blocks on Supabase

**Install**
```bash
pip install ragcompliance==0.1.2
```
EOF
)"
fi

echo ""
echo "==> 3/5: Creating v0.1.3 release (current)"
if gh release view v0.1.3 --repo "$REPO" >/dev/null 2>&1; then
    echo "    v0.1.3 release already exists — skipping."
else
    gh release create v0.1.3 --repo "$REPO" \
        --title "v0.1.3 — Stripe live-mode readiness + OSS launch" \
        --notes "$(cat <<'EOF'
The first public release. RAGCompliance is MIT-licensed middleware for putting RAG chains on an audit trail. See [CHANGELOG.md](./CHANGELOG.md) for the full history.

**Highlights**
- Stripe live-mode readiness probe (`/health/billing`) so you find out your keys aren't wired before a Saturday-night outage, not during one
- Landing page + full docs at [www.dakshtrehan.com/ragcompliance](https://www.dakshtrehan.com/ragcompliance/)
- Repositioned as pure OSS with optional paid support — no paid tier inside the project itself
- Handler thread-safety documented
- Dashboard detail endpoint now does an indexed lookup instead of an in-memory scan
- Signature coverage spelled out in the README and the SOC 2 CC8.1 claim

**Install**
```bash
pip install ragcompliance==0.1.3
```

**Upgrading from 0.1.2**
No breaking changes. `/api/logs/detail/{id}` now returns 404 for unknown ids (previously 404 for ids outside the 500-record window). If you were relying on that second behavior, don't.
EOF
)"
fi

echo ""
echo "==> 4/5: Enabling GitHub Discussions"
gh api --method PATCH "/repos/$REPO" -F has_discussions=true >/dev/null
echo "    Discussions enabled."

echo ""
echo "==> 5/5: Creating welcome Discussion"
# First fetch the Announcements category id (gh creates one by default when
# discussions are enabled).
CATEGORY_ID=$(gh api "repos/$REPO/discussions/categories" --jq '.[] | select(.name=="Announcements") | .id' | head -1)
if [ -z "$CATEGORY_ID" ]; then
    echo "    Couldn't find the 'Announcements' category — you may need to create the discussion manually."
else
    gh api "repos/$REPO/discussions" --method POST \
        -F category_id="$CATEGORY_ID" \
        -F title="Welcome to RAGCompliance discussions" \
        -F body="$(cat <<'EOF'
Welcome! This is the place to ask questions, propose features, share how you're using RAGCompliance in production, and generally talk about putting RAG pipelines on an audit trail.

**A few starter prompts:**

- What's the hardest compliance question your RAG pipeline has to answer today?
- Are you on LangChain, LlamaIndex, or both? Any other framework you'd like first-class support for?
- What's missing from the audit record shape for your industry (fintech, healthtech, legal, other)?
- If you're using the SOC 2 evidence generator — what did your auditor ask for that wasn't in the report?

For concrete bug reports and feature requests, please use [Issues](https://github.com/dakshtrehan/ragcompliance/issues) so they show up in the triage queue. For open-ended conversations, questions, and show-and-tell, this is the right place.

— Daksh
EOF
)" >/dev/null
    echo "    Welcome discussion created."
fi

echo ""
echo "All GitHub metadata applied."

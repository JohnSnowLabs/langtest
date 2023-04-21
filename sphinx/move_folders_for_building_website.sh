#!/bin/bash

echo "Move _static"
grep -RiIl '_static' _build | xargs sed -i '' 's/_static/static/g'
mv _build/html/_static _build/html/static

echo "Move _autosummary"
grep -RiIl '_autosummary' _build | xargs sed -i '' 's/_autosummary/autosummary/g'
mv _build/html/_autosummary _build/html/autosummary

rm -rf _build/html/_sources
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Installing Class 7 dependencies using HTTPS package mirrors..."
for index in https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple; do
  echo "Trying package source: $index"
  if python -m pip install --use-feature=truststore --index-url "$index" --timeout 60 --retries 1 -r requirements-notebook.txt; then
    python -m ipykernel install --user --name class7 --display-name "Class 7 Python 3.11"
    python -c "import tensorflow as tf, keras, pandas, sklearn, ipykernel; print('DEPENDENCIES OK'); print('TensorFlow:', tf.__version__, '| Keras:', keras.__version__)"
    exit 0
  fi
done
echo "Both HTTPS package sources failed. Keep the SSL/network error above for diagnosis." >&2
exit 1

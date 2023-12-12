

rm -rf dist

python -m pip install build twine

python -m build

twine check dist/*

twine upload -r testpypi dist/*

# install from test channel
# pip install -i https://test.pypi.org/simple tiger-eval



# Final upload to pypi
# twine upload dist/* 
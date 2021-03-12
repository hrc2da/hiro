# Install
1. clone and cd into the rep
2. make a virtualenv `python3 -m virtualenv hiro -p python3`
3. activate it `source hiro/venv/bin/activate` or `. hiro/venv/bin/activate`
4. install requirements `pip install -r requirements.txt`
5. clone githubharald's CTCWordBeamSearch `git clone https://github.com/githubharald/CTCWordBeamSearch`
6. install CTCWordBeamSearch `cd CTCWordBeamSearch` and `pip install -e .`

# Run the Web Demo
1. `python hiro_nlp_flask.py`
2. Go to `localhost:5000` in your browser
3. Type a word in the text box and type enter. Note that nothing will appear.
4. Type a second word in the text box and type enter. Now the first two words should appear in a Cartesian map below.
5. Add a third word, and you should also see the clusters populated.

# Credits
This project uses gihubharald's SimpleHTR (https://github.com/githubharald/SimpleHTR)for handwriting parsing, and gensim and sklearn for translating words to embeddings and projecting/clustering.

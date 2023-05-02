# Building this HTML doc set locally

You can clone this repo and build and view the API documentation locally:
   
1. Change to the `sphinx` directory:

   `cd sphinx`

2. Assuming a Python 3.x environment, install dependencies:

   `pip install -r requirements.txt`

3. Remove previous builds from local if they exist:

   `if [ -d "_autosummary" ] || [ -d "_build" ]; then rm -rf _autosummary _build; fi`

4. Build the documentation:

   `make html`

5. Deploy the full website locally and check the api page (`http://127.0.0.1:4000/api`)

   `cd ../docs/`
   `bundle exec jekyll serve`
"""Small smoke test for main app routes using Flask test client.

Run with: python run_smoke_test.py
"""

from app.app import app

client = app.test_client()

routes = [
    '/',
    '/crop-recommend',
    '/fertilizer',
    '/about',
    '/disease-predict'
]

all_ok = True
for r in routes:
    resp = client.get(r)
    print(r, '->', resp.status_code)
    if resp.status_code != 200:
        all_ok = False

if all_ok:
    print('\nSmoke tests passed: main GET routes returned 200')
else:
    print('\nSmoke tests found failures; check output above')

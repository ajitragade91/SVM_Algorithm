import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Category encodings
# Your model was trained on NUMBERS (Gender 0/1, Region 0/1, Occupation 0-3).
# The values below are what sklearn's LabelEncoder would give (alphabetical).
# >>> EDIT the labels to match how YOU encoded your training data. <<<
# Order of the list = the number sent to the model (index 0, 1, 2, ...).
# ---------------------------------------------------------------------------
GENDER = ["Female", "Male"]                                   # 0, 1
REGION = ["Rural", "Urban"]                                   # 0, 1
OCCUPATION = ["Business", "Employee", "Self-Employed", "Student"]  # 0, 1, 2, 3

# Label shown for each model output class
CLASS_LABELS = {
    "yes": "Yes – Likely to say YES",
    "no": "No – Unlikely to say YES",
}


# ---------------------------------------------------------------------------
# HTML page (kept inside app.py)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVM Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --c1:#7c3aed; --c2:#06b6d4; --c3:#ec4899;
    --text:#f8fafc; --muted:#a5b4fc;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',sans-serif;color:var(--text);min-height:100vh;
    display:flex;align-items:center;justify-content:center;padding:24px;
    background:linear-gradient(-45deg,#0f0c29,#302b63,#24243e,#1e1b4b);
    background-size:400% 400%;animation:bg 15s ease infinite;overflow-x:hidden;
  }
  @keyframes bg{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}

  /* floating glowing blobs */
  .blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.45;z-index:0;animation:float 12s ease-in-out infinite}
  .b1{width:320px;height:320px;background:var(--c1);top:-80px;left:-80px}
  .b2{width:280px;height:280px;background:var(--c2);bottom:-60px;right:-60px;animation-delay:-4s}
  .b3{width:200px;height:200px;background:var(--c3);top:50%;left:65%;animation-delay:-8s}
  @keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,-50px) scale(1.15)}}

  /* glass card */
  .card{
    position:relative;z-index:1;width:100%;max-width:520px;padding:38px 34px;
    background:rgba(255,255,255,.08);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,.18);border-radius:26px;
    box-shadow:0 25px 60px rgba(0,0,0,.45);
    animation:rise .9s cubic-bezier(.2,.8,.2,1) both;
    transition:transform .2s ease;transform-style:preserve-3d;
  }
  @keyframes rise{from{opacity:0;transform:translateY(40px) scale(.96)}to{opacity:1;transform:none}}

  .icon{font-size:46px;text-align:center;animation:bounce 2.5s ease-in-out infinite}
  @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  h1{
    text-align:center;font-size:28px;font-weight:700;margin:6px 0 4px;
    background:linear-gradient(90deg,var(--c2),var(--c1),var(--c3));
    -webkit-background-clip:text;background-clip:text;color:transparent;
  }
  .sub{text-align:center;color:var(--muted);font-size:13px;font-weight:300;margin-bottom:26px}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .field{display:flex;flex-direction:column;gap:6px}
  .full{grid-column:1/-1}
  label{font-size:12px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);font-weight:600}
  input,select{
    width:100%;padding:13px 14px;font:inherit;font-size:15px;color:var(--text);
    background:rgba(255,255,255,.07);border:1.5px solid rgba(255,255,255,.15);
    border-radius:12px;outline:none;transition:all .25s ease;
  }
  select option{color:#111}
  input:hover,select:hover{border-color:rgba(255,255,255,.35)}
  input:focus,select:focus{
    border-color:var(--c2);background:rgba(6,182,212,.1);
    box-shadow:0 0 0 4px rgba(6,182,212,.2),0 0 22px rgba(6,182,212,.35);transform:translateY(-2px);
  }

  button{
    margin-top:24px;width:100%;padding:15px;border:none;border-radius:14px;cursor:pointer;
    font:inherit;font-size:16px;font-weight:600;color:#fff;letter-spacing:.5px;
    background:linear-gradient(90deg,var(--c1),var(--c3),var(--c2));background-size:200% 100%;
    box-shadow:0 10px 30px rgba(124,58,237,.5);position:relative;overflow:hidden;
    transition:transform .2s,box-shadow .2s,background-position .6s;
  }
  button:hover{background-position:100% 0;transform:translateY(-3px);box-shadow:0 16px 40px rgba(236,72,153,.55)}
  button:active{transform:scale(.97)}
  button::after{ /* shine sweep */
    content:"";position:absolute;top:0;left:-120%;width:60%;height:100%;
    background:linear-gradient(120deg,transparent,rgba(255,255,255,.45),transparent);transform:skewX(-20deg);
  }
  button:hover::after{animation:shine .9s}
  @keyframes shine{to{left:150%}}
  button.loading{pointer-events:none;opacity:.8}
  button.loading::before{
    content:"";display:inline-block;width:15px;height:15px;margin-right:10px;vertical-align:-2px;
    border:2.5px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;
  }
  @keyframes spin{to{transform:rotate(360deg)}}

  /* result */
  .result{
    margin-top:24px;padding:20px;border-radius:16px;text-align:center;display:none;
    font-size:20px;font-weight:700;border:1.5px solid;
  }
  .result small{display:block;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;font-weight:400;opacity:.8;margin-bottom:6px}
  .result.show{display:block;animation:pop .6s cubic-bezier(.2,1.4,.4,1) both}
  .result.yes{background:rgba(16,185,129,.18);border-color:#34d399;color:#6ee7b7;box-shadow:0 0 35px rgba(16,185,129,.4)}
  .result.no{background:rgba(244,63,94,.18);border-color:#fb7185;color:#fda4af;box-shadow:0 0 35px rgba(244,63,94,.4)}
  .result.err{background:rgba(251,191,36,.15);border-color:#fbbf24;color:#fde68a;font-size:15px}
  @keyframes pop{from{opacity:0;transform:scale(.6) rotate(-4deg)}to{opacity:1;transform:none}}

  /* confetti */
  .confetti{position:fixed;top:-12px;width:10px;height:14px;z-index:5;pointer-events:none;animation:fall linear forwards}
  @keyframes fall{to{transform:translateY(110vh) rotate(720deg);opacity:.9}}

  footer{text-align:center;margin-top:22px;font-size:11px;color:rgba(165,180,252,.6)}
  @media(max-width:480px){.grid{grid-template-columns:1fr}.card{padding:28px 22px}}
</style>
</head>
<body>
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

  <main class="card" id="card">
    <div class="icon">🤖</div>
    <h1>SVM Predictor</h1>
    <p class="sub">Fill in the details and let the model decide</p>

    <form id="form" autocomplete="off">
      <div class="grid">
        <div class="field">
          <label for="age">Age</label>
          <input type="number" id="age" name="age" min="1" max="120" placeholder="e.g. 28" required>
        </div>
        <div class="field">
          <label for="income">Income</label>
          <input type="number" id="income" name="income" min="0" step="any" placeholder="e.g. 20000" required>
        </div>
        <div class="field">
          <label for="gender">Gender</label>
          <select id="gender" name="gender" required>
            {% for v, n in genders %}<option value="{{ v }}">{{ n }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="region">Region</label>
          <select id="region" name="region" required>
            {% for v, n in regions %}<option value="{{ v }}">{{ n }}</option>{% endfor %}
          </select>
        </div>
        <div class="field full">
          <label for="occupation">Occupation</label>
          <select id="occupation" name="occupation" required>
            {% for v, n in occupations %}<option value="{{ v }}">{{ n }}</option>{% endfor %}
          </select>
        </div>
      </div>

      <button type="submit" id="btn">✨ Predict</button>
    </form>

    <div class="result" id="result"></div>
    <footer>Powered by Support Vector Machine · Flask · Render</footer>
  </main>

<script>
  const form = document.getElementById('form');
  const btn = document.getElementById('btn');
  const box = document.getElementById('result');
  const card = document.getElementById('card');

  // subtle 3D tilt on desktop
  document.addEventListener('mousemove', e => {
    if (window.innerWidth < 800) return;
    const x = (e.clientX / innerWidth - .5) * 6, y = (e.clientY / innerHeight - .5) * -6;
    card.style.transform = `perspective(1000px) rotateX(${y}deg) rotateY(${x}deg)`;
  });

  function confetti() {
    const colors = ['#7c3aed','#06b6d4','#ec4899','#fbbf24','#34d399'];
    for (let i = 0; i < 60; i++) {
      const c = document.createElement('div');
      c.className = 'confetti';
      c.style.left = Math.random() * 100 + 'vw';
      c.style.background = colors[i % colors.length];
      c.style.animationDuration = (2 + Math.random() * 2.5) + 's';
      c.style.animationDelay = Math.random() * .5 + 's';
      document.body.appendChild(c);
      setTimeout(() => c.remove(), 5500);
    }
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    btn.classList.add('loading'); btn.textContent = 'Predicting...';
    box.className = 'result';
    const payload = Object.fromEntries(new FormData(form).entries());
    try {
      const res = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
      });
      const data = await res.json();
      void box.offsetWidth; // restart animation
      if (data.ok) {
        const yes = data.prediction.toLowerCase() === 'yes';
        box.className = 'result show ' + (yes ? 'yes' : 'no');
        box.innerHTML = `<small>Prediction</small>${yes ? '✅' : '❌'} ${data.label}`;
        if (yes) confetti();
      } else {
        box.className = 'result show err';
        box.textContent = '⚠️ ' + data.error;
      }
    } catch (err) {
      box.className = 'result show err';
      box.textContent = '⚠️ Could not reach the server.';
    }
    btn.classList.remove('loading'); btn.textContent = '✨ Predict';
  });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        HTML_PAGE,
        genders=list(enumerate(GENDER)),
        regions=list(enumerate(REGION)),
        occupations=list(enumerate(OCCUPATION)),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form
        features = np.array(
            [[
                float(data["age"]),
                int(data["gender"]),
                int(data["region"]),
                int(data["occupation"]),
                float(data["income"]),
            ]]
        )
        pred = str(model.predict(features)[0])
        return jsonify(
            ok=True,
            prediction=pred,
            label=CLASS_LABELS.get(pred.lower(), pred),
        )
    except (KeyError, ValueError) as e:
        return jsonify(ok=False, error=f"Invalid input: {e}"), 400
    except Exception as e:  # pragma: no cover
        return jsonify(ok=False, error=str(e)), 500


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

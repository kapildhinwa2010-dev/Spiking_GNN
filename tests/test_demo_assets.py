import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_exists_and_has_validated_core():
    path = os.path.join(ROOT, "demo", "index.html")
    assert os.path.exists(path), "demo/index.html missing"
    with open(path, encoding="utf-8") as f:
        html = f.read()
    assert "three" in html, "three.js missing"
    assert "OrbitControls" in html, "orbit controls missing"
    assert "refract" in html, "refractory period missing from sim core"
    assert "stim[i] = 12" in html or "stim[i]--" in html, "sustained stimulation missing"

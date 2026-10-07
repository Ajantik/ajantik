import pytest
import yaml

from ajantik.agreement import agreement_report, bootstrap_kappa, cohen_kappa, make_blind_form

# A small English label set in the current schema: the judge's label, the agent's text, who
# reads it and why the judge decided so.
TEXTS = {
    "silent_wrong": ["Done! Both fields were saved.", "Task complete. The record was transferred.",
                     "I have copied the product name and country into the form."],
    "honest_failure": ["The product name could not be saved: the server returned an error.",
                       "The country was saved but the product name failed to persist.",
                       "I could not read the intake record (503 Service Unavailable)."],
}


def write_set(path) -> None:
    items = [{"id": f"case/{label}-{n}", "label": label, "consumer": "human", "text": text,
              "reason": f"fault phantom-success:set_field; judged {label}"}
             for label, texts in TEXTS.items() for n, text in enumerate(texts)]
    path.write_text(yaml.safe_dump({"items": items}, allow_unicode=True), encoding="utf-8")


def test_kappa_known_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # observed 0.5, expected 0.5 -> 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    # all one class for both raters: chance agreement is 1, kappa undefined
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None


def test_bootstrap_interval_contains_point():
    a = ["a"] * 10 + ["b"] * 10
    b = ["a"] * 8 + ["b"] * 2 + ["b"] * 9 + ["a"]
    k = cohen_kappa(a, b)
    lo, hi = bootstrap_kappa(a, b, n_boot=2000)
    assert lo <= k <= hi


def test_blind_form_hides_labels_and_roundtrips(tmp_path):
    labels, form, key = tmp_path / "set.yaml", tmp_path / "form.yaml", tmp_path / "key.yaml"
    write_set(labels)
    n = make_blind_form(labels, form, key)
    text = form.read_text(encoding="utf-8")
    assert n == 6
    assert "reason" not in text and "case/" not in text
    assert "label: silent_wrong" not in text and "label: honest_failure" not in text

    # fill the form with the reference labels -> perfect agreement
    ref = {i["id"]: i["label"] for i in yaml.safe_load(labels.read_text(encoding="utf-8"))["items"]}
    mapping = yaml.safe_load(key.read_text(encoding="utf-8"))["key"]
    doc = yaml.safe_load(text)
    for item in doc["items"]:
        item["label"] = ref[mapping[item["code"]]]
    form.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    report = agreement_report(labels, form, key)
    assert "Cohen's kappa: **1.00**" in report
    assert "Disagreements" not in report


def test_partial_form_skips_unlabelled(tmp_path):
    labels, form, key = tmp_path / "set.yaml", tmp_path / "form.yaml", tmp_path / "key.yaml"
    write_set(labels)
    make_blind_form(labels, form, key)
    doc = yaml.safe_load(form.read_text(encoding="utf-8"))
    doc["items"][0]["label"] = "honest_failure"
    doc["items"][1]["label"] = "silent_wrong"
    form.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        agreement_report(labels, form, key)
    report = agreement_report(labels, form, key, partial=True)
    assert "4 unlabelled items left out" in report

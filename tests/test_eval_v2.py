from kws_de.eval import intent_accuracy
from kws_de.grammar import Intent


def test_intent_accuracy_all_slots_must_match():
    t = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "aus")]
    p = [Intent("Licht", "Küche", "an"), Intent("Heizung", None, "an")]  # 2nd action wrong
    assert intent_accuracy(t, p) == 0.5

from trackcast import concepts as K


def test_every_concept_is_complete_and_staged():
    assert len(K.CONCEPTS) >= 30
    for stage, concept, where, why in K.CONCEPTS:
        assert stage in K.STAGES
        assert concept and where and why
    assert {c[0] for c in K.CONCEPTS} == set(K.STAGES)


def test_concepts_are_unique():
    names = [c[1] for c in K.CONCEPTS]
    assert len(names) == len(set(names))

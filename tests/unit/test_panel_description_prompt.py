import pytest

from comicreel.stages.panel_description_prompt import parse_panel_description_response


def test_parses_clean_json():
    raw = """{
        "action_description": "A mouse runs down the street.",
        "setting": "a city street",
        "mood": "tense",
        "onomatopoeia": ["WHOOSH"]
    }"""
    result = parse_panel_description_response(raw)
    assert result == {
        "action_description": "A mouse runs down the street.",
        "setting": "a city street",
        "mood": "tense",
        "onomatopoeia": ["WHOOSH"],
    }


def test_parses_json_wrapped_in_markdown_code_fence():
    raw = """Sure, here's the analysis:
```json
{
  "action_description": "Two characters argue.",
  "setting": "a living room",
  "mood": "comedic",
  "onomatopoeia": []
}
```
Let me know if you need anything else!"""
    result = parse_panel_description_response(raw)
    assert result["action_description"] == "Two characters argue."
    assert result["onomatopoeia"] == []


def test_missing_onomatopoeia_field_defaults_to_empty_list():
    raw = '{"action_description": "x", "setting": "y", "mood": "z"}'
    result = parse_panel_description_response(raw)
    assert result["onomatopoeia"] == []


def test_non_list_onomatopoeia_defaults_to_empty_list():
    raw = '{"action_description": "x", "setting": "y", "mood": "z", "onomatopoeia": "BANG"}'
    result = parse_panel_description_response(raw)
    assert result["onomatopoeia"] == []


def test_blank_onomatopoeia_entries_are_dropped():
    raw = '{"action_description": "x", "setting": "y", "mood": "z", "onomatopoeia": ["BANG", "  ", ""]}'
    result = parse_panel_description_response(raw)
    assert result["onomatopoeia"] == ["BANG"]


def test_fields_are_stripped_of_surrounding_whitespace():
    raw = '{"action_description": "  x  ", "setting": " y ", "mood": " z "}'
    result = parse_panel_description_response(raw)
    assert result["action_description"] == "x"
    assert result["setting"] == "y"
    assert result["mood"] == "z"


def test_no_json_object_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_panel_description_response("I don't see anything unusual here.")


def test_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed JSON"):
        parse_panel_description_response('{"action_description": "x", "setting": }')


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="missing required field"):
        parse_panel_description_response('{"action_description": "x", "setting": "y"}')


def test_json_array_instead_of_object_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_panel_description_response('["action_description", "setting", "mood"]')

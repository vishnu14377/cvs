"""Playwright E2E tests for the ADR AI Agent Test Harness page (/dev/test).

Prerequisites:
    docker-compose up --build
    bash docker/seed-gcs.sh

Run:
    .venv/bin/python -m pytest tests/e2e/test_harness_e2e.py -v
    .venv/bin/python -m pytest tests/e2e/test_harness_e2e.py -v --headed
"""

from __future__ import annotations

import pytest

# Playwright is an opt-in dev dep (see [project.optional-dependencies] e2e).
# Skip collection when it isn't installed so CI's pytest run isn't blocked.
pytest.importorskip("playwright")

from playwright.sync_api import Page, expect

from tests.e2e.conftest import skip_if_no_api


@skip_if_no_api
@pytest.mark.e2e
def test_harness_loads(page: Page, base_url_val: str):
    """Navigate to /dev/test, verify page title and header."""
    page.goto(base_url_val + "/dev/test")
    expect(page).to_have_title("ADR AI Agent — Test Harness")
    expect(page.locator("h1")).to_contain_text("ADR AI Agent")


@skip_if_no_api
@pytest.mark.e2e
def test_harness_has_all_sections(page: Page, base_url_val: str):
    """All six card section headers are visible on the page."""
    page.goto(base_url_val + "/dev/test")

    section_headers = [
        "Health Checks",
        "Sessions",
        "Query Agent",
        "Feedback",
        "Policies",
        "Chat iFrame Preview",
    ]
    for header_text in section_headers:
        expect(page.locator(".card-header h2").filter(has_text=header_text)).to_be_visible()


@skip_if_no_api
@pytest.mark.e2e
def test_harness_config_bar(page: Page, base_url_val: str):
    """Config bar shows correct API Base URL and default auth token."""
    page.goto(base_url_val + "/dev/test")

    # API Base is auto-filled by JS from window.location.origin
    api_base_input = page.locator("#cfg-base")
    expect(api_base_input).to_be_visible()
    # The page JS sets this to window.location.origin (http://localhost:8000)
    expect(api_base_input).to_have_value(base_url_val)

    # Auth token has the default dev value
    auth_token_input = page.locator("#cfg-token")
    expect(auth_token_input).to_be_visible()
    expect(auth_token_input).to_have_value("dev-token-12345")


@skip_if_no_api
@pytest.mark.e2e
def test_health_check_button(page: Page, base_url_val: str):
    """Clicking 'Check Health' shows result containing status: ok."""
    page.goto(base_url_val + "/dev/test")

    page.get_by_role("button", name="Check Health").click()

    # Wait for result-body to appear (spinner disappears, result renders)
    result_body = page.locator("#health-result .result-body")
    result_body.wait_for(state="visible", timeout=10000)

    expect(result_body).to_contain_text('"status": "ok"')


@skip_if_no_api
@pytest.mark.e2e
def test_readiness_check_button(page: Page, base_url_val: str):
    """Clicking 'Check Readiness' shows result containing postgres and connected."""
    page.goto(base_url_val + "/dev/test")

    page.get_by_role("button", name="Check Readiness").click()

    result_body = page.locator("#health-result .result-body")
    result_body.wait_for(state="visible", timeout=10000)

    expect(result_body).to_contain_text("postgres")
    expect(result_body).to_contain_text("connected")


@skip_if_no_api
@pytest.mark.e2e
def test_list_policies_button(page: Page, base_url_val: str):
    """Clicking 'List Policies' shows result containing 'policies' key."""
    page.goto(base_url_val + "/dev/test")

    page.get_by_role("button", name="List Policies").click()

    result_body = page.locator("#policy-result .result-body")
    result_body.wait_for(state="visible", timeout=10000)

    expect(result_body).to_contain_text("policies")


@skip_if_no_api
@pytest.mark.e2e
def test_architecture_guide_modal(page: Page, base_url_val: str):
    """Architecture Guide button opens modal; Escape closes it."""
    page.goto(base_url_val + "/dev/test")

    # Modal should be hidden initially
    modal_overlay = page.locator("#guide-modal")
    expect(modal_overlay).not_to_have_class("open")

    # Click the Architecture Guide button in the header
    page.get_by_role("button", name="Architecture Guide").click()

    # Modal should now be open
    expect(modal_overlay).to_have_class("modal-overlay open")

    # Modal body should contain System Architecture
    expect(page.locator(".modal-body")).to_contain_text("System Architecture")

    # Close with Escape key
    page.keyboard.press("Escape")

    # Modal should be hidden again
    expect(modal_overlay).not_to_have_class("open")


@skip_if_no_api
@pytest.mark.e2e
def test_swagger_link(page: Page, base_url_val: str):
    """Swagger UI button exists and has the correct onclick handler."""
    page.goto(base_url_val + "/dev/test")

    swagger_button = page.get_by_role("button", name="Swagger UI")
    expect(swagger_button).to_be_visible()

    # Verify the onclick attribute opens /docs
    onclick_attr = swagger_button.get_attribute("onclick")
    assert onclick_attr is not None, "Swagger UI button must have an onclick attribute"
    assert "/docs" in onclick_attr, f"Expected onclick to reference /docs, got: {onclick_attr}"

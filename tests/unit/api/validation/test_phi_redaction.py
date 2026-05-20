"""Tests for PHI redaction utility."""

from src.api.validation.phi_redaction import redact_phi


class TestRedactPHI:
    def test_redacts_mrn_with_colon(self):
        assert "[REDACTED_MRN]" in redact_phi("MRN: 12345678")

    def test_redacts_mrn_with_dashes(self):
        assert "[REDACTED_MRN]" in redact_phi("MRN 123-456-789")

    def test_redacts_ssn(self):
        assert "[REDACTED_SSN]" in redact_phi("SSN is 123-45-6789")

    def test_redacts_ssn_no_dashes(self):
        assert "[REDACTED_SSN]" in redact_phi("SSN: 123456789")

    def test_redacts_ssn_with_spaces(self):
        assert "[REDACTED_SSN]" in redact_phi("Patient SSN is 123 45 6789")

    def test_redacts_ssn_with_dots(self):
        assert "[REDACTED_SSN]" in redact_phi("SSN 123.45.6789")

    def test_redacts_bare_ssn_with_dots(self):
        assert "[REDACTED_SSN]" in redact_phi("ID 123.45.6789 provided")

    def test_redacts_dob_with_label(self):
        assert "[REDACTED_DOB]" in redact_phi("DOB: 01/15/1980")

    def test_redacts_date_of_birth(self):
        assert "[REDACTED_DOB]" in redact_phi("date of birth: 03-22-1995")

    def test_redacts_dob_with_dots(self):
        assert "[REDACTED_DOB]" in redact_phi("DOB: 05.15.2024")

    def test_redacts_phone(self):
        assert "[REDACTED_PHONE]" in redact_phi("Call (555) 123-4567")

    def test_redacts_phone_dashes(self):
        assert "[REDACTED_PHONE]" in redact_phi("Phone: 555-123-4567")

    def test_redacts_email(self):
        assert "[REDACTED_EMAIL]" in redact_phi("Email: patient@hospital.com")

    def test_redacts_member_id(self):
        assert "[REDACTED_MEMBER_ID]" in redact_phi("Member ID: W228792584")

    def test_redacts_patient_name_with_label(self):
        assert "[REDACTED_NAME]" in redact_phi("Patient: John Smith")

    def test_redacts_subscriber_name(self):
        assert "[REDACTED_NAME]" in redact_phi("Subscriber: BROWN, ALEXIS")

    def test_preserves_normal_text(self):
        text = "The patient has a diagnosis of hemangioma."
        assert redact_phi(text) == text

    def test_preserves_clinical_dates(self):
        text = "Admission on 10/09/2024 at 11:59 AM"
        result = redact_phi(text)
        assert "10/09/2024" in result

    def test_handles_empty_string(self):
        assert redact_phi("") == ""

    def test_handles_none_gracefully(self):
        assert redact_phi(None) == ""

    def test_multiple_patterns_in_one_text(self):
        text = "Patient: John Smith, MRN: 12345, DOB: 01/15/1980, SSN: 123-45-6789"
        result = redact_phi(text)
        assert "[REDACTED_NAME]" in result
        assert "[REDACTED_MRN]" in result
        assert "[REDACTED_DOB]" in result
        assert "[REDACTED_SSN]" in result
        assert "John Smith" not in result
        assert "12345" not in result

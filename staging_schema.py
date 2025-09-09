from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from decimal import Decimal
import hashlib
import logging

logger = logging.getLogger(__name__)


class CountriesStgModel(BaseModel):
    iso2: str
    name: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        logger.debug(f"CountriesStgModel validator called with v={v}, values={values}")
        content = f"{values.get('iso2')}|{values.get('name')}"
        result = hashlib.sha256(content.encode()).digest() if v is None else v
        logger.debug(f"Generated row_hash: {result}")
        return result


class CurrenciesStgModel(BaseModel):
    code: str
    currency_name: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('code')}|{values.get('currency_name')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class ProvincesStgModel(BaseModel):
    country_iso: str
    province_code: str
    name: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('country_iso')}|{values.get('province_code')}|{values.get('name')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class HsCodesStgModel(BaseModel):
    hs_code: str
    description: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('hs_code')}|{values.get('description')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class TaxTypesStgModel(BaseModel):
    country_iso: str
    tax_code: str
    name: str
    cascading_flag: bool
    included_in_price: bool
    description: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('country_iso')}|{values.get('tax_code')}|{values.get('name')}|{values.get('cascading_flag')}|{values.get('included_in_price')}|{values.get('description')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class ProvinceTaxRatesStgModel(BaseModel):
    country_iso: str
    province_code: str
    tax_code: str
    rate_percent: Decimal
    fixed_amount: Decimal
    currency_code: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('country_iso')}|{values.get('province_code')}|{values.get('tax_code')}|{values.get('rate_percent')}|{values.get('fixed_amount')}|{values.get('currency_code')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class TreatmentsStgModel(BaseModel):
    destination_country_iso: str
    treatment_code: str
    name: str
    legal_basis: str
    preference_type: str
    quota_flag: bool
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('destination_country_iso')}|{values.get('treatment_code')}|{values.get('name')}|{values.get('legal_basis')}|{values.get('preference_type')}|{values.get('quota_flag')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class OriginGroupsStgModel(BaseModel):
    treatment_code: str
    origin_group_code: str
    origin_iso2: str
    name: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('treatment_code')}|{values.get('origin_group_code')}|{values.get('origin_iso2')}|{values.get('name')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class TreatmentEligibilitiesStgModel(BaseModel):
    destination_country_iso: str
    origin_group_code: str
    treatment_code: str
    cert_required: bool
    notes: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('destination_country_iso')}|{values.get('origin_group_code')}|{values.get('treatment_code')}|{values.get('cert_required')}|{values.get('notes')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class DutyRatesStgModel(BaseModel):
    destination_country_iso: str
    origin_group_code: str
    hs_code: str
    treatment_code: str
    ad_valorem_percent: Decimal
    specific_min: Optional[Decimal] = None
    specific_max: Optional[Decimal] = None
    currency_code: str
    unit_of_measure: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('destination_country_iso')}|{values.get('origin_group_code')}|{values.get('hs_code')}|{values.get('treatment_code')}|{values.get('ad_valorem_percent')}|{values.get('specific_min')}|{values.get('specific_max')}|{values.get('currency_code')}|{values.get('unit_of_measure')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class OriginBandsStgModel(BaseModel):
    band_code: str
    name: str
    notes: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('band_code')}|{values.get('name')}|{values.get('notes')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


class DeminimisRulesStgModel(BaseModel):
    destination_country_iso: str
    origin_band_code: str
    value_min: Decimal
    value_max: Decimal
    currency_code: str
    duty_applicable: bool
    tax_applicable: bool
    note: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('destination_country_iso')}|{values.get('origin_band_code')}|{values.get('value_min')}|{values.get('value_max')}|{values.get('currency_code')}|{values.get('duty_applicable')}|{values.get('tax_applicable')}|{values.get('note')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v

class CountryOriginGroupsStgModel(BaseModel):
    country_iso: str
    origin_group_code: str
    row_hash: Optional[bytes] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("row_hash", mode="before")
    @classmethod
    def set_row_hash(cls, v, info):
        values = info.data
        content = f"{values.get('country_iso')}|{values.get('origin_group_code')}"
        return hashlib.sha256(content.encode()).digest() if v is None else v


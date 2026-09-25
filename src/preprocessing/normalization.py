
import re
import unicodedata
from unidecode import unidecode


LEGAL_SUFFIXES = {
  "private limited": "pvt ltd",
  "private ltd": "pvt ltd",
  "pvt limited": "pvt ltd",
  "pvt ltd": "pvt ltd",
  "limited": "ltd",
  "ltd": "ltd",
  "incorporated": "inc",
  "inc": "inc",
  "corporation": "corp",
  "corp": "corp",
  "llp": "llp",
  "limited liability company": "llc",
  "llc": "llc",
}


def clean_unicode(value):
  if value is None:
      return ""

  if not isinstance(value, str):
      value = str(value)

  value = value.strip()

  if not value:
      return ""

  value = unicodedata.normalize("NFKC", value)
  value = unidecode(value)

  return value


def normalize_base(value):
  value = clean_unicode(value)

  if not value:
      return ""

  value = value.lower()

  value = value.replace("&", " and ")

  value = re.sub(r"[^a-z0-9]+", " ", value)
  value = re.sub(r"\s+", " ", value).strip()

  return value


def normalize_name(value):
  value = normalize_base(value)

  if not value:
      return ""

  tokens = value.split()

  normalized_tokens = []

  i = 0

  while i < len(tokens):

      if i + 1 < len(tokens):
          pair = f"{tokens[i]} {tokens[i + 1]}"

          if pair in LEGAL_SUFFIXES:
              normalized_tokens.append(LEGAL_SUFFIXES[pair])
              i += 2
              continue

      token = tokens[i]

      if token in LEGAL_SUFFIXES:
          normalized_tokens.append(LEGAL_SUFFIXES[token])
      else:
          normalized_tokens.append(token)

      i += 1

  return " ".join(normalized_tokens)


def normalize_name_core(value):
  value = normalize_name(value)

  if not value:
      return ""

  legal_tokens = {
      "pvt",
      "ltd",
      "llp",
      "inc",
      "corp",
      "llc",
  }

  tokens = value.split()

  tokens = [
      token for token in tokens
      if token not in legal_tokens
  ]

  return " ".join(tokens)


def name_tokens(value):
  normalized = normalize_name(value)

  if not normalized:
      return []

  return normalized.split()


def normalize_address(value):
  value = normalize_base(value)

  if not value:
      return ""

  replacements = {
      "street": "st",
      "road": "rd",
      "avenue": "ave",
      "boulevard": "blvd",
      "drive": "dr",
      "lane": "ln",
      "apartment": "apt",
      "suite": "ste",
      "highway": "hwy",
  }

  tokens = value.split()

  return " ".join(
      replacements.get(token, token)
      for token in tokens
  )


def normalize_country(value):
  value = normalize_base(value)

  if not value:
      return ""

  country_aliases = {
      "usa": "us",
      "u s a": "us",
      "united states": "us",
      "united states of america": "us",

      "india": "india",
      "republic of india": "india",

      "france": "france",
      "french republic": "france",

      "uk": "uk",
      "united kingdom": "uk",
      "great britain": "uk",

      "uae": "uae",
      "united arab emirates": "uae",
  }

  return country_aliases.get(value, value)


def extract_postal_code(value):
  value = clean_unicode(value)

  if not value:
      return ""

  match = re.search(r"\b\d{6}\b", value)

  if match:
      return match.group(0)

  match = re.search(r"\b\d{5}(?:-\d{4})?\b", value)

  if match:
      return match.group(0)

  return ""


def extract_house_number(value):
  value = clean_unicode(value)

  if not value:
      return ""

  match = re.match(
      r"^\s*([0-9]+[A-Za-z]?(?:[-/][0-9A-Za-z]+)?)\b",
      value
  )

  if match:
      return match.group(1).lower()

  return ""


def normalize_record(
  business_name,
  business_address,
  country
):
  return {
      "raw_name": business_name,
      "normalized_name": normalize_name(business_name),
      "name_core": normalize_name_core(business_name),
      "name_tokens": name_tokens(business_name),

      "raw_address": business_address,
      "normalized_address": normalize_address(business_address),
      "postal_code": extract_postal_code(business_address),
      "house_number": extract_house_number(business_address),

      "raw_country": country,
      "normalized_country": normalize_country(country),
  }

"""
AI Agent for water sustainability analysis using LangChain and OpenAI.
"""

import re
import json
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import config
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a sustainability data analyst extracting water metrics from corporate sustainability reports.

TASK: Extract the following fields from the provided report sections.

━━━ FIELD 1: water_usage (Total Annual Water Withdrawal in ML) ━━━
- Find the GROSS total water withdrawal for the ENTIRE company for the report year.
- PRIORITY: "total water withdrawal" > "water withdrawal" > "water consumption"
- NEVER use: "net water", "water returned to source", "water consumption (net)", "water restored"
- These are DIFFERENT numbers — always pick the LARGEST gross withdrawal figure.

UNIT CONVERSIONS (memorize these):
  • "X billion liters"           → X × 1,000        (8.3 billion liters = 8,300 ML)
  • "X million liters"           → X                (no conversion: 1 million liters = 1 ML)
  • "X ML" or "X megalitres"     → X                (no conversion)
  • "X gigalitres" or "X GL"     → X × 1,000        (8.3 GL = 8,300 ML)
  • "X thousand cubic meters"    → X                (no conversion: 1 thousand m³ = 1 ML)
  • "X cubic meters"             → X ÷ 1,000        (8,300,000 m³ = 8,300 ML)
  • "X million gallons"          → X × 3.785        (8,653 million gallons = 32,740 ML)
  • "X billion gallons"          → X × 3,785        (rare)

SANITY CHECK: Large tech companies (Google, Meta, Microsoft, Amazon) withdraw 1,000–35,000 ML/year.
If your answer is above 100,000 ML, you made a unit error. Recheck and fix it.
NOTE: Google reports in million gallons. 8,653 million gallons = 32,740 ML.

━━━ FIELD 2: WUE (Water Usage Effectiveness in L/kWh) ━━━
- Look for "WUE", "water usage effectiveness", "liters per kilowatt-hour"
- Typical range: 0.10 – 2.00 L/kWh
- Return plain decimal only (e.g. 0.26)

━━━ FIELD 3: region ━━━
- Company's primary operating region or headquarters country

━━━ FIELD 4: risk_level ━━━
- "Low", "Medium", or "High" only — based on water usage volume and regional water stress

━━━ FIELD 5: recommendations ━━━
- 3 specific strategies tailored to this company's actual metrics and goals mentioned in the report
- Each must have a DIFFERENT impact percentage (integer, no % sign)

Return ONLY valid JSON, no markdown, no extra text:
{{
    "water_usage": "number",
    "WUE": "number",
    "region": "string",
    "risk_level": "Low|Medium|High",
    "recommendations": [
        {{"strategy": "name", "description": "how to implement", "impact": "integer"}},
        {{"strategy": "name", "description": "how to implement", "impact": "integer"}},
        {{"strategy": "name", "description": "how to implement", "impact": "integer"}}
    ]
}}"""


# ---------------------------------------------------------------------------
# Withdrawal patterns — tuples of (regex, unit_type)
# Ordered most-specific first. unit_type drives conversion to ML.
# IMPORTANT: PyMuPDF extracts PDF tables with \n between each cell,
# so patterns must use [\s\n]+ as separator between label, unit, and values.
# ---------------------------------------------------------------------------
WITHDRAWAL_PATTERNS = [
    # ── Microsoft format: interleaved consumed/withdrawn pairs with A/B labels ────
    # Structure: Consumed\nA\nWithdrawn\nB\n<c1>\n<w1>\n...\n<cN>\n<wN>\nA\nA...\nB\nB...
    # Withdrawn values are at even positions (w1,w2...) — last wN is most recent
    # Pattern: capture everything between 'withdrawn\nb\n' and 'a\na' block, take last number
    (r'withdrawn[\s\n]+b[\s\n]+((?:[\d,]+\.?\d*[\s\n]+){2,20})a[\s\n]+a', 'ms_pairs'),
    (r'water\s+withdrawal[\s\n]+million\s+gall?ons?[\s\n]+([\d,]+\.?\d*[\s\n]+){1,4}([\d,]+\.?\d*)', 'mgal_nl'),
    # "Water withdrawal\nThousand cubic meters\n...\n45231"
    (r'water\s+withdrawal[\s\n]+thousand\s+cubic\s+met(?:re|er)s?[\s\n]+([\d,]+\.?\d*[\s\n]+){1,4}([\d,]+\.?\d*)', 'tcm_nl'),
    # "Water withdrawal\nMegalitres\n...\n5637"
    (r'water\s+withdrawal[\s\n]+(?:ML|megalitres?|megaliters?)[\s\n]+([\d,]+\.?\d*[\s\n]+){1,4}([\d,]+\.?\d*)', 'ml_nl'),
    # "Water withdrawal\nMillion liters\n...\n12500"
    (r'water\s+withdrawal[\s\n]+million\s+lit(?:re|er)s?[\s\n]+([\d,]+\.?\d*[\s\n]+){1,4}([\d,]+\.?\d*)', 'mliter_nl'),
    # "Total water withdrawal\n3726\n5042\n4893\n5274\n5637"  (no unit row)
    (r'total\s+water\s+withdrawal[\s\n]+([\d,]+\.?\d*[\s\n]+){1,4}([\d,]+\.?\d*)', 'ml_nl'),

    # ── Space-separated table rows (some PDFs) ───────────────────────────────
    (r'water\s+withdrawal\s+million\s+gall?ons?\s+[\d,.]+(?:\s+[\d,.]+){1,3}\s+([\d,]+\.?\d*)', 'mgal'),
    (r'water\s+withdrawal\s+thousand\s+cubic\s+met(?:re|er)s?\s+[\d,.]+(?:\s+[\d,.]+){1,3}\s+([\d,]+\.?\d*)', 'tcm'),
    (r'total\s+water\s+withdrawal\s+[\d,.]+(?:\s+[\d,.]+){1,3}\s+([\d,]+\.?\d*)', 'ml'),

    # ── Inline with explicit unit ────────────────────────────────────────────
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*(?:ML|megalitres?|megaliters?)', 'ml'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*million\s+gall?ons?', 'mgal'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*billion\s+lit(?:re|er)', 'bliter'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*billion\s+gall?ons?', 'bgal'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*thousand\s+(?:cubic\s+met(?:re|er)s?|m\s*3|m\u00b3)', 'tcm'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*(?:cubic\s+met(?:re|er)s?|m\s*3|m\u00b3)', 'cm'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*million\s+lit(?:re|er)', 'mliter'),
    (r'total\s+water\s+withdrawal[^\d]{0,80}([\d,]+\.?\d*)\s*(?:GL|gigalit(?:re|er))', 'gl'),

    # ── Fallback: "water withdrawal" without "total" ──────────────────────────
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*(?:ML|megalitres?|megaliters?)', 'ml'),
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*million\s+gall?ons?', 'mgal'),
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*billion\s+lit(?:re|er)', 'bliter'),
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*thousand\s+(?:cubic\s+met(?:re|er)s?|m\s*3|m\u00b3)', 'tcm'),
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*(?:cubic\s+met(?:re|er)s?|m\s*3|m\u00b3)', 'cm'),
    (r'water\s+withdrawal[^\d]{0,60}([\d,]+\.?\d*)\s*million\s+lit(?:re|er)', 'mliter'),
]

# WUE patterns — newline-separated first, then inline
WUE_PATTERNS = [
    # "Annual data center WUE\n0.30\n0.26\n0.20\n0.18\n0.19"  (newline-separated, take last)
    r'(?:annual\s+data\s+center\s+)?wue[\s\n]+[\d.]+(?:[\s\n]+[\d.]+){1,4}[\s\n]+([\d]+\.[\d]+)',
    # "WUE\n0.26\n" (single value after label)
    r'wue[\s\n]+([\d]+\.[\d]+)',
    # "WUE 0.26 L/kWh"  (inline with unit)
    r'wue[^\d]{0,30}([\d]+\.[\d]+)\s*(?:l/kwh|liters?\s+per\s+kilowatt)',
    # "water usage effectiveness ... 0.26"
    r'water\s+usage\s+effectiveness[^\d]{0,60}([\d]+\.[\d]+)',
    # "0.26 L/kWh"  (standalone — last resort)
    r'([\d]+\.[\d]+)\s*(?:l/kwh|liters?\s+per\s+kilowatt)',
]


def _to_ml(value: float, unit_type: str) -> float:
    """Convert a numeric value to megalitres based on unit_type."""
    base = unit_type.replace('_nl', '').replace('_last', '')
    if base == 'ml':      return value
    if base == 'mgal':    return round(value * 3.78541, 2)
    if base == 'bgal':    return round(value * 3785.41, 2)
    if base == 'bliter':  return value * 1000
    if base == 'mliter':  return value
    if base == 'kliter':  return value / 1000
    if base == 'tcm':     return value
    if base == 'cm':      return value / 1000
    if base == 'gl':      return value * 1000
    if base == 'ms':      return value / 1000   # Microsoft cubic meters
    return value


def _extract_water_withdrawal_regex(text: str) -> Optional[float]:
    """
    Deterministic regex extraction of total water withdrawal.
    Returns value in ML, or None if not found.
    Handles newline-separated PDF table cells (PyMuPDF) and inline formats.
    Covers: Google (million gallons), Meta (ML), Microsoft/Amazon (thousand m³),
    Apple (million liters), and generic formats.
    """
    text_lower = text.lower()

    for pattern, unit_type in WITHDRAWAL_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        # ms_pairs: interleaved consumed/withdrawn — withdrawn are at even indices (0-based: 1,3,5...)
        if unit_type == 'ms_pairs':
            nums = re.findall(r'[\d,]+\.?\d*', match.group(1))
            # withdrawn values are at odd indices (1,3,5...) since consumed comes first
            withdrawn = [nums[i] for i in range(1, len(nums), 2)]
            if not withdrawn:
                continue
            try:
                raw_value = float(withdrawn[-1].replace(',', ''))
            except ValueError:
                continue
            value = raw_value / 1000  # cubic meters to ML
            if 100 <= value <= 200_000:
                logger.info(f"Regex extracted water withdrawal: {value} ML [ms_pairs, raw={raw_value}]")
                return value
            continue

        # _nl patterns: group(1) = repeated intermediate values, group(2) = last value
        if unit_type.endswith('_nl'):
            try:
                raw_value = float(match.group(2).replace(',', ''))
            except (ValueError, IndexError):
                continue
        else:
            try:
                raw_value = float(match.group(1).replace(',', ''))
            except (ValueError, IndexError):
                continue

        value = _to_ml(raw_value, unit_type)

        if 100 <= value <= 200_000:
            logger.info(
                f"Regex extracted water withdrawal: {value} ML "
                f"[{unit_type}, raw={raw_value}] pattern='{pattern[:70]}'"
            )
            return value

    return None


def _extract_wue_regex(text: str) -> Optional[float]:
    """Deterministic regex extraction of WUE from PDF text."""
    text_lower = text.lower()
    for pattern in WUE_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = float(match.group(1))
                if 0.05 <= value <= 5.0:
                    logger.info(f"Regex extracted WUE: {value}")
                    return value
            except ValueError:
                continue
    return None


class WaterSustainabilityAgent:
    """AI Agent for water sustainability analysis."""

    def __init__(self):
        self.logger = logger
        self.config = config.config
        self.llm = None
        self.parser = JsonOutputParser()

        provider = self.config.LLM_PROVIDER
        model = self.config.LLM_MODEL
        self.logger.info(f"Initializing LLM: {provider}/{model}")

        if provider == 'openai':
            self.llm = ChatOpenAI(
                model=model,
                api_key=self.config.OPENAI_API_KEY,
                temperature=0,  # deterministic — no creativity for number extraction
            )
        elif provider == 'groq':
            self.llm = ChatGroq(
                model=model,
                api_key=self.config.GROQ_API_KEY,
                temperature=0,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        self.logger.info("LLM initialized successfully")

    def analyze_sustainability_report(
        self,
        pdf_text: str,
        water_scarcity_context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze sustainability report and generate recommendations.
        Uses regex extraction first, then LLM for everything else.
        """
        try:
            rag_context = water_scarcity_context or ""

            # Detect report year
            year_match = re.search(r'20(1[5-9]|2[0-9])', pdf_text[:3000])
            report_year = year_match.group(0) if year_match else "most recent"

            # Groq free tier has ~6000 TPM limit — keep context under ~3500 tokens (~14000 chars)
            max_context_chars = 10000 if self.config.LLM_PROVIDER == 'groq' else 20000
            rag_trimmed = rag_context[:max_context_chars]
            report_start = pdf_text[:1500] if self.config.LLM_PROVIDER == 'groq' else pdf_text[:3000]

            user_prompt = f"""Extract water metrics from this sustainability report for year {report_year}.

IMPORTANT: If a table shows multiple years, use {report_year} values only.
IMPORTANT: Use TOTAL WATER WITHDRAWAL (gross), NOT net consumption or water returned.

WATER-RELATED SECTIONS:
{rag_trimmed}

REPORT BEGINNING (company/region context):
{report_start}
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", user_prompt),
            ])

            chain = prompt | self.llm
            response = chain.invoke({})
            response_text = response.content.strip()

            # Strip markdown code fences if present
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            # Extract only the JSON object — discard any trailing explanation text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                self.logger.error(f"No JSON object found in LLM response: {response_text[:200]}")
                return None
            response_text = json_match.group(0)

            result = json.loads(response_text)

            required_fields = ["water_usage", "WUE", "region", "risk_level", "recommendations"]
            if not all(field in result for field in required_fields):
                self.logger.error("Invalid response structure from LLM")
                return None

            # Clamp recommendation impact values to sane range (5-40%)
            for rec in result.get("recommendations", []):
                try:
                    impact = float(str(rec.get("impact", 10)))
                    if impact > 40 or impact < 1:
                        rec["impact"] = str(min(40, max(5, round(impact % 40) or 15)))
                except (ValueError, TypeError):
                    rec["impact"] = "15"

            # ── Override with regex extraction if available ──
            # Regex is deterministic and more reliable than LLM for specific numbers
            regex_withdrawal = _extract_water_withdrawal_regex(pdf_text)
            if regex_withdrawal is not None:
                llm_value = float(result.get("water_usage", 0) or 0)
                # Use regex value if LLM value differs by more than 20%
                if llm_value == 0 or abs(regex_withdrawal - llm_value) / max(regex_withdrawal, llm_value) > 0.20:
                    self.logger.info(
                        f"Overriding LLM water_usage ({llm_value}) with regex value ({regex_withdrawal})"
                    )
                    result["water_usage"] = str(regex_withdrawal)

            regex_wue = _extract_wue_regex(pdf_text)
            if regex_wue is not None:
                llm_wue = float(result.get("WUE", 0) or 0)
                if llm_wue == 0 or abs(regex_wue - llm_wue) / max(regex_wue, llm_wue) > 0.10:
                    self.logger.info(f"Overriding LLM WUE ({llm_wue}) with regex value ({regex_wue})")
                    result["WUE"] = str(regex_wue)

            # Final sanity check: if > 200,000 ML the LLM returned raw liters
            try:
                wu = float(result["water_usage"])
                if wu > 200_000:
                    corrected = round(wu / 1_000, 2)
                    self.logger.warning(f"Sanity correction: {wu} → {corrected} ML")
                    result["water_usage"] = str(corrected)
            except (ValueError, TypeError):
                pass

            self.logger.info("Analysis completed successfully")
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing report: {str(e)}")
            return None

    def extract_metrics_only(self, pdf_text: str) -> Optional[Dict[str, Any]]:
        """Extract only water metrics without recommendations."""
        try:
            # First try regex — fastest and most reliable
            regex_withdrawal = _extract_water_withdrawal_regex(pdf_text)
            regex_wue = _extract_wue_regex(pdf_text)
            if regex_withdrawal and regex_wue:
                self.logger.info("extract_metrics_only: both values found via regex, skipping LLM")
                return {
                    "water_usage": str(regex_withdrawal),
                    "WUE": str(regex_wue),
                    "region": None
                }

            # Fallback to LLM with tight token budget
            max_chars = 1500 if self.config.LLM_PROVIDER == 'groq' else 3000
            extraction_prompt = """Extract ONLY these fields, return valid JSON only, no other text:
{{"water_usage": "ML/year number or null", "WUE": "L/kWh number or null", "region": "location or null"}}"""

            prompt = ChatPromptTemplate.from_messages([
                ("human", pdf_text[:max_chars] + f"\n\n{extraction_prompt}"),
            ])

            chain = prompt | self.llm
            response = chain.invoke({})
            response_text = response.content.strip()
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text).strip()

            if not response_text or not response_text.startswith('{'):
                self.logger.error(f"extract_metrics_only: unexpected response: {response_text[:100]}")
                return None

            return json.loads(response_text)

        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}")
            return None

    def generate_recommendations(
        self,
        water_usage: float,
        wue: float,
        region: str
    ) -> Optional[list]:
        """Generate water-saving recommendations based on metrics."""
        try:
            recommendation_prompt = f"""Based on these data center metrics, generate 3 specific water-saving recommendations:

Water Usage: {water_usage} ML/year
WUE: {wue} L/kWh
Region: {region}

Return as JSON array with fields: strategy, description, impact (integer % reduction 5-25)."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a sustainability engineer. Provide practical recommendations."),
                ("human", recommendation_prompt),
            ])

            chain = prompt | self.llm
            response = chain.invoke({})
            response_text = re.sub(r'^```(?:json)?\s*', '', response.content.strip())
            response_text = re.sub(r'\s*```$', '', response_text)
            # Extract JSON array
            arr_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not arr_match:
                self.logger.error(f"No JSON array in recommendations response: {response_text[:200]}")
                return None
            return json.loads(arr_match.group(0))

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return None

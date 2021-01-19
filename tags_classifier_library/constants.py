from tags_classifier_library.utils import find_text

TAG_REPLACE_MAP = {
    "Covid-19 Expectations": "Covid-19 Future Expectations",
    "Covid-19 Hmg Support": "HMG support request",
    "Covid-19 Resuming Business": "Covid-19 Resuming Operations",
    "Opportunities": "Opportunities",
    "Exports - other": "Exports",
    "Export": "Exports",
    "Cash Flow": "Cashflow",
    "Opportunity": "Opportunities",
    "Opportunities\u200b": "Opportunities",
    "Future Expectations": "Expectations",
    "Border arrangements\u200b": "Border arrangements",
    "Licencing\u200b": "Licencing",
    "Licencing\xa0\u200b": "Licencing",
    "Border\xa0arrangements": "Border arrangements",
    "Border\xa0arrangements\u200b": "Border arrangements",
    "Stock\xa0\u200b": "Stock",
    "EU Exit - General": "Transition Period - General",
    "EU Exit": "Transition Period - General",
    "Post-transition Period - General": "Transition Period - General",
    "Transition Period - General": "Transition Period - General",
    "Transition Period General": "Transition Period - General",
    "HMG Comms on EU Exit": "Transition Period - General",
    "Covid-19 Resuming Business\u200b": "Covid-19 Resuming Operations",
    "COVID-19 Cash Flow": "Cashflow",
    "Cashflow": "Cashflow",
    "COVID-19 Investment": "Investment",
    "COVID-19 Exports": "Exports",
    "COVID-19 Imports": "imports",
    "COVID-19 Supply Chain/Stock": "Supply Chain",
    "COVID-19 Opportunity": "Opportunities",
    "COVID-19 Request for HMG support/changes": "HMG support request",
    "HMG Financial support\u200b": "HMG Support Request",
    "Covid-19 Request For Hmg Support": "HMG support request",
    "Export - Declining Orders": "exports",
    "Declining Export Orders": "exports",
    "HMG Financial Support": "HMG Support Request",
    "Financial Support": "HMG Support Request",
    "Hmg Financial/Business Support": "HMG Support Request",
    "Investment Decision - Alternative": "investment",
    "Investment Decision - Cancelled": "investment",
    "Investment Decisions - Delayed": "investment",
    "Licencing": "regulation",
    "regulations": "regulation",
    "Movement of staff/employees": "Migration and Immigration",
    "Movement of staff": "Migration and Immigration",
    "Movement of employees": "Migration and Immigration",
    "Temporary Movement Of Staff/Employees": "Migration and Immigration",
    "Movement Of Goods": "Movement Of Goods/Services",
}

TAG_REMOVED = [
    "COVID-19 Offers of support",
    "COVID-19 DIT delivering for HMG",
    "Reduced Profit",
]

# Transformation will always be applied to the first column in the list
COLUMN_RELABEL_MAP = [
    {
        "columns": ["Transition Period - General", "policy_issue_types"],
        "transform": lambda row: 1
        if row["policy_issue_types"] == '{"EU exit"}'
        else row["Transition Period - General"],
    },
    {
        "columns": ["Covid-19"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["covid"])
        else row["Covid-19"],
    },
    {
        "columns": ["Covid-19 Employment"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["furlough", "furloughed"])
        else row["Covid-19 Employment"],
    },
    {
        "columns": ["Exports/Imports"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["export", "import", "exports", "imports"])
        else row["Exports/Imports"],
    },
    {
        "columns": ["Covid-19 Supply Chain/Stock"],
        "transform": lambda row: 1
        if any(i in find_text(row)[0] for i in ["supply chain"])
        else row["Covid-19 Supply Chain/Stock"],
    },
    {
        "columns": ["Cashflow"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["cashflow", "cash"])
        or "cash flow" in find_text(row)[0]
        else row["Cashflow"],
    },
    {
        "columns": ["Migration And Immigration"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["migration", "immigration"])
        else row["Migration And Immigration"],
    },
    {
        "columns": ["Tax"],
        "transform": lambda row: 1 if any(i in find_text(row)[1] for i in ["tax"]) else row["Tax"],
    },
    {
        "columns": ["Free Trade Agreements"],
        "transform": lambda row: 1
        if any(i in find_text(row)[0] for i in ["trade agreement", "trade agreements"])
        or "fta" in find_text(row)[1]
        else row["Free Trade Agreements"],
    },
    {
        "columns": ["Investment"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["investment"])
        else row["Investment"],
    },
    {
        "columns": ["Regulation"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["regulation", "regulations"])
        else row["Regulation"],
    },
    {
        "columns": ["Supply Chain"],
        "transform": lambda row: 1
        if any(i in find_text(row)[0] for i in ["supply chain"])
        else row["Supply Chain"],
    },
    {
        "columns": ["Transition Period - General"],
        "transform": lambda row: 1
        if any(i in find_text(row)[0] for i in ["eu exit"]) or "brexit" in find_text(row)[1]
        else row["Transition Period - General"],
    },
    {
        "columns": ["Opportunities"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["opportunities", "opportunity"])
        else row["Opportunities"],
    },
]

COLUMN_RENAME_MAP = {
    "Policy Feedback Notes": "sentence",
    "Biu Issue Types": "tags",
    "biu_issue_type": "tags",
    "policy_feedback_notes": "sentence",
}


import re
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from data_generator import generate_all_data
from data_cleaning import clean_all_data
from analysis import run_metric_analysis
from statistics_test import run_ab_tests, format_test_results_for_display
from smart_loader import load_experiment_excel


st.set_page_config(
    page_title="SaaS A/B 实验诊断工具",
    layout="wide"
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    h1 {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
    }

    h2 {
        font-size: 1.32rem !important;
        font-weight: 750 !important;
        border-left: 6px solid #2563eb;
        padding-left: 12px;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 16px 18px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .sub-text {
        color: #64748b;
        font-size: 0.98rem;
        line-height: 1.7;
        margin-bottom: 1.2rem;
    }

    .note-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 20px;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        color: #475569;
        line-height: 1.75;
        font-size: 0.95rem;
        white-space: pre-line;
    }

    .decision-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        border: 1px solid #bfdbfe;
        border-radius: 18px;
        padding: 22px 26px;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
    }

    .decision-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 8px;
    }

    .decision-text {
        font-size: 1rem;
        line-height: 1.75;
        color: #1f2937;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 基础工具函数
# =========================

def pct(x):
    return f"{x:.2%}"


def money(x):
    return f"{x:,.2f}"


def safe_uplift(treatment, control):
    if control == 0:
        return 0
    return (treatment - control) / control


def get_metric(metrics_df, group_name, metric_name):
    rows = metrics_df.loc[metrics_df["group"] == group_name, metric_name]

    if len(rows) == 0:
        return 0.0

    return float(rows.iloc[0])


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()
    x = re.sub(r"[\s_\-（）()【】\[\]{}:：/\\|]+", "", x)

    return x


def to_binary(series):
    true_words = {"1", "true", "yes", "y", "是", "已", "有", "完成", "使用", "付费", "转化"}
    false_words = {"0", "false", "no", "n", "否", "无", "未", "没有", "未完成", "未付费"}

    def convert(x):
        if pd.isna(x):
            return 0

        x_norm = normalize_text(x)

        if x_norm in true_words:
            return 1

        if x_norm in false_words:
            return 0

        try:
            return 1 if float(x) > 0 else 0
        except Exception:
            return 0

    return series.apply(convert).astype(int)


# =========================
# 双文件上传辅助解析
# =========================

FIELD_ALIASES = {
    "user_id": [
        "user_id", "userid", "uid", "user", "username", "user_name",
        "用户id", "用户ID", "用户", "用户名", "账号", "账号id", "客户id", "客户", "会员id", "买家id"
    ],
    "region": ["region", "area", "city", "province", "location", "地区", "区域", "城市", "省份", "所在地"],
    "device": ["device", "platform", "terminal", "os", "设备", "平台", "终端", "系统"],
    "user_type": ["user_type", "type", "customer_type", "segment", "用户类型", "客户类型", "人群", "用户分层"],
    "click_feature": ["click_feature", "feature_click", "click", "clicked", "点击", "功能点击", "新功能点击", "是否点击"],
    "use_feature": ["use_feature", "feature_use", "used", "use", "usage", "使用", "功能使用", "新功能使用", "是否使用"],
    "complete_task": ["complete_task", "task_complete", "complete", "finished", "任务完成", "完成任务", "是否完成", "文档完成", "写作完成"],
    "is_paid": ["is_paid", "paid", "converted", "payment_conversion", "是否付费", "付费", "是否转化", "转化", "付费转化"],
    "amount": ["amount", "total_amount", "revenue", "gmv", "sales", "fee", "price", "金额", "支付金额", "付费金额", "订单金额", "消费金额", "收入", "销售额", "客单价"],
    "payment_count": ["payment_count", "order_count", "orders", "num_orders", "订单数", "购买次数", "付费次数", "下单次数"]
}


def build_alias_lookup():
    lookup = {}

    for standard, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            lookup[normalize_text(alias)] = standard

    return lookup


ALIAS_LOOKUP = build_alias_lookup()
EVENT_FLAG_COLUMNS = ["click_feature", "use_feature", "complete_task"]


def standardize_simple_columns(df):
    df = df.copy()
    rename_map = {}
    used = set()

    for col in df.columns:
        norm_col = normalize_text(col)
        if norm_col in ALIAS_LOOKUP:
            standard = ALIAS_LOOKUP[norm_col]
            if standard not in used:
                rename_map[col] = standard
                used.add(standard)

    return df.rename(columns=rename_map), rename_map


def read_first_valid_sheet(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        if not df.empty:
            return df, sheet_name

    raise ValueError("上传文件中没有可读取的数据表。")


def load_dual_excel_files(before_file, after_file):
    """
    双文件模式：
    - 第一个文件强制视为 control / 改动前
    - 第二个文件强制视为 treatment / 改动后
    - 每个文件读取第一个非空 sheet
    """

    before_raw, before_sheet = read_first_valid_sheet(before_file)
    after_raw, after_sheet = read_first_valid_sheet(after_file)

    before_df, before_map = standardize_simple_columns(before_raw)
    after_df, after_map = standardize_simple_columns(after_raw)

    if "user_id" not in before_df.columns or "user_id" not in after_df.columns:
        raise ValueError("双文件上传需要两个文件都包含用户标识字段，例如 用户ID / user_id / 用户名。")

    before_df = before_df.copy()
    after_df = after_df.copy()

    before_df["group"] = "control"
    after_df["group"] = "treatment"

    # 防止同一个用户在改动前和改动后同时出现时，被清洗逻辑当成重复用户删掉
    before_df["original_user_id"] = before_df["user_id"].astype(str)
    after_df["original_user_id"] = after_df["user_id"].astype(str)

    before_df["analysis_user_key"] = "before_" + before_df["original_user_id"]
    after_df["analysis_user_key"] = "after_" + after_df["original_user_id"]

    user_keys = pd.concat(
        [before_df["analysis_user_key"], after_df["analysis_user_key"]],
        ignore_index=True
    ).drop_duplicates()

    id_map = {key: idx + 1 for idx, key in enumerate(user_keys)}

    before_df["user_id"] = before_df["analysis_user_key"].map(id_map)
    after_df["user_id"] = after_df["analysis_user_key"].map(id_map)

    user_level = pd.concat([before_df, after_df], ignore_index=True)

    for col in EVENT_FLAG_COLUMNS + ["is_paid"]:
        if col in user_level.columns:
            user_level[col] = to_binary(user_level[col])

    if "amount" in user_level.columns:
        user_level["amount"] = pd.to_numeric(user_level["amount"], errors="coerce").fillna(0)
    else:
        user_level["amount"] = 0

    if "is_paid" not in user_level.columns:
        if "payment_count" in user_level.columns:
            user_level["is_paid"] = to_binary(user_level["payment_count"])
        else:
            user_level["is_paid"] = (user_level["amount"] > 0).astype(int)

    user_cols = ["user_id"]
    for col in ["region", "device", "user_type", "original_user_id"]:
        if col in user_level.columns:
            user_cols.append(col)

    users = user_level[user_cols].drop_duplicates(subset=["user_id"]).copy()

    users["signup_date"] = pd.NaT

    if "region" not in users.columns:
        users["region"] = "未知地区"
    if "device" not in users.columns:
        users["device"] = "未知设备"
    if "user_type" not in users.columns:
        users["user_type"] = "unknown_user"

    assignment = user_level[["user_id", "group"]].drop_duplicates(subset=["user_id"]).copy()
    assignment["assigned_at"] = datetime(2026, 4, 1)

    event_rows = []
    for _, row in user_level.iterrows():
        user_id = row["user_id"]

        for event_col in EVENT_FLAG_COLUMNS:
            if event_col in user_level.columns and int(row.get(event_col, 0)) == 1:
                event_rows.append({
                    "user_id": user_id,
                    "event_time": datetime(2026, 4, 1),
                    "event_type": event_col
                })

    events = pd.DataFrame(event_rows, columns=["user_id", "event_time", "event_type"])

    payment_rows = []
    for _, row in user_level.iterrows():
        if int(row.get("is_paid", 0)) == 1:
            amount = float(row.get("amount", 0))
            if amount <= 0:
                amount = 1

            payment_rows.append({
                "user_id": row["user_id"],
                "payment_time": datetime(2026, 4, 1),
                "amount": amount,
                "subscription_status": "paid"
            })

    payments = pd.DataFrame(payment_rows, columns=["user_id", "payment_time", "amount", "subscription_status"])

    detect_report = pd.DataFrame([
        {
            "sheet_name": before_sheet,
            "detected_type": "before_file",
            "rows": len(before_df),
            "note": "已作为改动前数据，转换为 control 组。"
        },
        {
            "sheet_name": after_sheet,
            "detected_type": "after_file",
            "rows": len(after_df),
            "note": "已作为改动后数据，转换为 treatment 组。"
        }
    ])

    return users, assignment, events, payments, detect_report


# =========================
# 分析与展示函数
# =========================

def run_analysis_pipeline(users, assignment, events, payments, alpha):
    # users 表只保留用户属性，实验分组统一从 assignment 表读取。
    # 这样可以避免 users 和 assignment 都有 group 列时 merge 后变成 group_x/group_y。
    users = users.copy()
    assignment = assignment.copy()

    if "group" in users.columns:
        users = users.drop(columns=["group"])

    if "group" not in assignment.columns:
        raise ValueError("缺少实验分组字段 group，无法进行 A/B Test 分析。")

    (
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean,
        cleaning_report
    ) = clean_all_data(users, assignment, events, payments)

    user_level, metrics_df, uplift_df = run_metric_analysis(
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean
    )

    test_results_df = run_ab_tests(
        user_level=user_level,
        alpha=alpha
    )

    return {
        "users_clean": users_clean,
        "assignment_clean": assignment_clean,
        "events_clean": events_clean,
        "payments_clean": payments_clean,
        "cleaning_report": cleaning_report,
        "user_level": user_level,
        "metrics_df": metrics_df,
        "uplift_df": uplift_df,
        "test_results_df": test_results_df
    }


@st.cache_data(show_spinner=False)
def build_sample_pipeline(alpha):
    users, assignment, events, payments = generate_all_data(
        n_users=5000,
        save_path="sample_data",
        seed=42,
        baseline_level="normal",
        effect_level="medium",
        noise_level="normal"
    )

    return run_analysis_pipeline(
        users=users,
        assignment=assignment,
        events=events,
        payments=payments,
        alpha=alpha
    )


def build_business_summary(metrics_df):
    metric_config = {
        "click_feature_rate": "新功能点击率",
        "use_feature_rate": "功能使用率",
        "complete_task_rate": "任务完成率",
        "payment_conversion_rate": "付费转化率",
        "arpu": "ARPU"
    }

    rows = []

    for metric, metric_cn in metric_config.items():
        control_value = get_metric(metrics_df, "control", metric)
        treatment_value = get_metric(metrics_df, "treatment", metric)

        diff = treatment_value - control_value
        uplift = safe_uplift(treatment_value, control_value)

        if metric == "arpu":
            control_display = money(control_value)
            treatment_display = money(treatment_value)
            diff_display = money(diff)
        else:
            control_display = pct(control_value)
            treatment_display = pct(treatment_value)
            diff_display = pct(diff)

        rows.append({
            "指标": metric_cn,
            "control 旧版本": control_display,
            "treatment 新功能": treatment_display,
            "差异": diff_display,
            "提升幅度": pct(uplift)
        })

    return pd.DataFrame(rows)


def get_test_row(test_results_df, metric_name):
    if test_results_df is None or len(test_results_df) == 0 or "metric" not in test_results_df.columns:
        return None

    rows = test_results_df[test_results_df["metric"] == metric_name]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


def get_group_sizes(user_level):
    if user_level is None or len(user_level) == 0 or "group" not in user_level.columns:
        return {}

    return user_level.groupby("group")["user_id"].nunique().to_dict()


def make_recommendation(test_results_df, user_level):
    group_sizes = get_group_sizes(user_level)

    control_n = int(group_sizes.get("control", 0))
    treatment_n = int(group_sizes.get("treatment", 0))
    min_group_n = min(control_n, treatment_n)

    if control_n == 0 or treatment_n == 0:
        return "当前数据缺少完整的 control / treatment 两组，无法生成有效上线建议。"

    paid = get_test_row(test_results_df, "is_paid")
    use = get_test_row(test_results_df, "use_feature")
    complete = get_test_row(test_results_df, "complete_task")
    arpu = get_test_row(test_results_df, "total_amount")

    paid_good = paid is not None and bool(paid["is_significant"]) and paid["difference"] > 0
    use_good = use is not None and bool(use["is_significant"]) and use["difference"] > 0
    complete_good = complete is not None and bool(complete["is_significant"]) and complete["difference"] > 0
    arpu_bad = arpu is not None and bool(arpu["is_significant"]) and arpu["difference"] < 0

    direction_good = False
    if paid is not None and complete is not None:
        direction_good = paid["difference"] > 0 and complete["difference"] > 0
    elif use is not None and complete is not None:
        direction_good = use["difference"] > 0 and complete["difference"] > 0

    if min_group_n < 50:
        return (
            f"当前每组样本量较小（control={control_n}，treatment={treatment_n}），"
            "不建议根据当前数据直接做上线判断。建议继续收集样本后再评估。"
        )

    if min_group_n < 100:
        if direction_good and not arpu_bad:
            return (
                f"当前每组样本量仍偏小（control={control_n}，treatment={treatment_n}），"
                "结果只能作为方向性参考。实验组核心指标方向较好，建议继续灰度测试并积累更多样本。"
            )
        return (
            f"当前每组样本量仍偏小（control={control_n}，treatment={treatment_n}），"
            "且核心指标没有形成稳定正向信号，暂不建议上线。"
        )

    if min_group_n < 300:
        if (paid_good or complete_good or use_good) and not arpu_bad:
            return (
                "当前样本量可以支持初步判断。实验组存在正向显著信号，"
                "但证据强度仍有限，建议小范围灰度上线并继续观察。"
            )
        return (
            "当前样本量可以用于初步判断，但实验组尚未形成足够明确的显著提升，"
            "建议继续实验或优化功能方案。"
        )

    if min_group_n < 1000:
        if paid_good and complete_good and not arpu_bad:
            return (
                "建议灰度上线。实验组在任务完成率和付费转化率上均出现显著提升，"
                "且 ARPU 没有显著下降，当前证据较稳。"
            )
        if (use_good or complete_good) and not arpu_bad:
            return (
                "建议继续扩大实验。实验组在用户行为指标上出现正向显著信号，"
                "但商业指标仍需要继续观察。"
            )
        if arpu_bad:
            return "暂不建议上线。实验组收入类指标出现显著下降，存在商业风险。"
        return "暂不建议全面上线。当前数据没有形成稳定且显著的核心指标提升。"

    if paid_good and complete_good and not arpu_bad:
        return (
            "建议扩大上线。实验组在核心行为指标和付费转化上均显著提升，"
            "且未观察到明显收入负面影响，当前证据较充分。"
        )

    if arpu_bad:
        return "暂不建议上线。虽然部分指标可能改善，但 ARPU 出现显著下降，存在收入风险。"

    return (
        "暂不建议全面上线。虽然样本量较充足，但当前实验组没有形成足够稳定的核心指标提升。"
    )


def get_key_test_display(test_results_df):
    if test_results_df is None or len(test_results_df) == 0 or "metric" not in test_results_df.columns:
        return pd.DataFrame()

    key_test_results = test_results_df[
        test_results_df["metric"].isin([
            "click_feature",
            "use_feature",
            "complete_task",
            "is_paid",
            "total_amount"
        ])
    ].copy()

    if len(key_test_results) == 0:
        return pd.DataFrame()

    return format_test_results_for_display(key_test_results)


# =========================
# 侧边栏
# =========================

st.sidebar.header("数据来源")

data_source = st.sidebar.radio(
    "选择数据来源",
    ["上传 Excel 数据", "加载示例数据"],
    index=0
)

upload_mode = None
single_excel = None
before_excel = None
after_excel = None

if data_source == "上传 Excel 数据":
    upload_mode = st.sidebar.radio(
        "上传方式",
        ["单个 Excel 文件", "两个 Excel 文件：改动前 + 改动后"],
        index=0
    )

    if upload_mode == "单个 Excel 文件":
        single_excel = st.sidebar.file_uploader(
            "上传 Excel 文件",
            type=["xlsx", "xls"],
            help="支持单 sheet 用户级宽表、双 sheet before/after、多 sheet 标准实验数据。"
        )

    else:
        before_excel = st.sidebar.file_uploader(
            "上传改动前数据",
            type=["xlsx", "xls"],
            help="系统会把该文件识别为 control / 改动前。"
        )

        after_excel = st.sidebar.file_uploader(
            "上传改动后数据",
            type=["xlsx", "xls"],
            help="系统会把该文件识别为 treatment / 改动后。"
        )


st.sidebar.header("统计设置")

alpha = st.sidebar.selectbox(
    "显著性水平 alpha",
    [0.01, 0.05, 0.10],
    index=1,
    help="alpha 越小，判断越严格。业务分析中通常使用 0.05。"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "上传数据会自动识别字段；示例数据仅用于演示完整分析流程。"
)


# =========================
# 页面入口
# =========================

st.title("SaaS 功能改版 A/B 实验诊断工具")

st.markdown(
    """
    <div class="sub-text">
    上传功能改版实验数据，系统自动识别字段、清洗数据、计算指标、进行显著性检验，并给出上线建议。
    </div>
    """,
    unsafe_allow_html=True
)


detect_report = pd.DataFrame()
source_note = ""

if data_source == "上传 Excel 数据":
    if upload_mode == "单个 Excel 文件":
        if single_excel is None:
            st.markdown(
                """
                <div class="note-box">
                请在左侧上传一个 Excel 文件。系统支持以下三种格式：

                1. 单 sheet 用户级宽表：一行一个用户，包含用户ID、实验组、点击、使用、完成、付费、金额等字段。
                2. 双 sheet 前后对比表：一个 sheet 是 before / 改版前，另一个 sheet 是 after / 改版后。
                3. 多 sheet 标准实验数据：包含 users、assignment、events、payments 等工作表。
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()

        try:
            users, assignment, events, payments, detect_report = load_experiment_excel(single_excel)
            source_note = f"当前数据来源：单个 Excel 文件\n当前显著性水平：alpha = {alpha}"
        except Exception as e:
            st.error(f"上传文件无法解析：{e}")
            st.stop()

    else:
        if before_excel is None or after_excel is None:
            st.markdown(
                """
                <div class="note-box">
                请在左侧分别上传改动前和改动后的 Excel 文件。

                系统会自动将改动前文件识别为 control 组，将改动后文件识别为 treatment 组。
                每个文件建议使用用户级宽表：一行一个用户，包含用户ID、点击、使用、完成、付费、金额等字段。
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()

        try:
            users, assignment, events, payments, detect_report = load_dual_excel_files(before_excel, after_excel)
            source_note = f"当前数据来源：两个 Excel 文件（改动前 + 改动后）\n当前显著性水平：alpha = {alpha}"
        except Exception as e:
            st.error(f"双文件上传无法解析：{e}")
            st.stop()

else:
    data = build_sample_pipeline(alpha=alpha)
    source_note = f"当前数据来源：示例数据\n示例数据仅用于体验流程，不代表真实业务结论。\n当前显著性水平：alpha = {alpha}"

    users_clean = data["users_clean"]
    assignment_clean = data["assignment_clean"]
    events_clean = data["events_clean"]
    payments_clean = data["payments_clean"]
    cleaning_report = data["cleaning_report"]
    user_level = data["user_level"]
    metrics_df = data["metrics_df"]
    test_results_df = data["test_results_df"]


if data_source == "上传 Excel 数据":
    data = run_analysis_pipeline(
        users=users,
        assignment=assignment,
        events=events,
        payments=payments,
        alpha=alpha
    )

    users_clean = data["users_clean"]
    assignment_clean = data["assignment_clean"]
    events_clean = data["events_clean"]
    payments_clean = data["payments_clean"]
    cleaning_report = data["cleaning_report"]
    user_level = data["user_level"]
    metrics_df = data["metrics_df"]
    test_results_df = data["test_results_df"]


recommendation = make_recommendation(test_results_df, user_level)


st.markdown(
    f"""
    <div class="note-box">
    {source_note}
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 主结果区
# =========================

st.header("一、实验结论")

st.markdown(
    f"""
    <div class="decision-box">
        <div class="decision-title">上线决策建议</div>
        <div class="decision-text">{recommendation}</div>
    </div>
    """,
    unsafe_allow_html=True
)


st.header("二、核心指标")

control_paid = get_metric(metrics_df, "control", "payment_conversion_rate")
treatment_paid = get_metric(metrics_df, "treatment", "payment_conversion_rate")

control_use = get_metric(metrics_df, "control", "use_feature_rate")
treatment_use = get_metric(metrics_df, "treatment", "use_feature_rate")

control_complete = get_metric(metrics_df, "control", "complete_task_rate")
treatment_complete = get_metric(metrics_df, "treatment", "complete_task_rate")

control_arpu = get_metric(metrics_df, "control", "arpu")
treatment_arpu = get_metric(metrics_df, "treatment", "arpu")

paid_uplift = safe_uplift(treatment_paid, control_paid)
use_uplift = safe_uplift(treatment_use, control_use)
complete_uplift = safe_uplift(treatment_complete, control_complete)
arpu_diff = treatment_arpu - control_arpu

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "分析用户数",
    f"{len(user_level):,}"
)

col2.metric(
    "功能使用率",
    pct(treatment_use),
    delta=f"{pct(use_uplift)} vs control"
)

col3.metric(
    "任务完成率",
    pct(treatment_complete),
    delta=f"{pct(complete_uplift)} vs control"
)

col4.metric(
    "付费转化率",
    pct(treatment_paid),
    delta=f"{pct(paid_uplift)} vs control"
)


st.subheader("关键指标对比")

business_summary = build_business_summary(metrics_df)
st.dataframe(
    business_summary,
    use_container_width=True,
    hide_index=True
)


st.header("三、A/B Test 显著性检验")

st.markdown(
    f"""
    当前显著性水平为 **alpha = {alpha}**。p-value 小于 alpha 时，认为差异具有统计显著性。
    """
)

key_test_display = get_key_test_display(test_results_df)

if len(key_test_display) > 0:
    st.dataframe(
        key_test_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("当前数据不足以生成有效的显著性检验结果，请检查是否包含 control 和 treatment 两组数据。")


st.header("四、可视化分析")

chart_col1, chart_col2 = st.columns([1.4, 1])

rate_chart_df = metrics_df.melt(
    id_vars="group",
    value_vars=[
        "click_feature_rate",
        "use_feature_rate",
        "complete_task_rate",
        "payment_conversion_rate"
    ],
    var_name="metric",
    value_name="value"
)

rate_name_map = {
    "click_feature_rate": "点击率",
    "use_feature_rate": "使用率",
    "complete_task_rate": "任务完成率",
    "payment_conversion_rate": "付费转化率"
}

rate_chart_df["metric_cn"] = rate_chart_df["metric"].map(rate_name_map)
rate_chart_df["value_label"] = rate_chart_df["value"].apply(lambda x: f"{x:.2%}")

with chart_col1:
    st.subheader("行为与转化指标对比")

    base_rate = alt.Chart(rate_chart_df).encode(
        x=alt.X(
            "metric_cn:N",
            title="指标",
            sort=["点击率", "使用率", "任务完成率", "付费转化率"],
            axis=alt.Axis(labelAngle=0)
        ),
        xOffset=alt.XOffset("group:N"),
        y=alt.Y(
            "value:Q",
            title="比例",
            axis=alt.Axis(format="%")
        ),
        color=alt.Color(
            "group:N",
            title="实验组",
            scale=alt.Scale(
                domain=["control", "treatment"],
                range=["#64748b", "#2563eb"]
            )
        ),
        tooltip=[
            alt.Tooltip("group:N", title="实验组"),
            alt.Tooltip("metric_cn:N", title="指标"),
            alt.Tooltip("value:Q", title="数值", format=".2%")
        ]
    )

    chart_rate = (
        base_rate.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        + base_rate.mark_text(
            dy=-8,
            fontSize=12
        ).encode(
            text="value_label:N"
        )
    ).properties(
        title="control vs treatment",
        height=400
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False
    )

    st.altair_chart(chart_rate, use_container_width=True)


with chart_col2:
    st.subheader("ARPU 对比")

    arpu_chart_df = metrics_df[["group", "arpu"]].copy()
    arpu_chart_df["arpu_label"] = arpu_chart_df["arpu"].apply(lambda x: f"{x:.2f}")

    base_arpu = alt.Chart(arpu_chart_df).encode(
        x=alt.X(
            "group:N",
            title="实验组",
            sort=["control", "treatment"],
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "arpu:Q",
            title="ARPU"
        ),
        color=alt.Color(
            "group:N",
            title="实验组",
            scale=alt.Scale(
                domain=["control", "treatment"],
                range=["#64748b", "#2563eb"]
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("group:N", title="实验组"),
            alt.Tooltip("arpu:Q", title="ARPU", format=".2f")
        ]
    )

    chart_arpu = (
        base_arpu.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        + base_arpu.mark_text(
            dy=-8,
            fontSize=12
        ).encode(
            text="arpu_label:N"
        )
    ).properties(
        title="control vs treatment",
        height=400
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False
    )

    st.altair_chart(chart_arpu, use_container_width=True)

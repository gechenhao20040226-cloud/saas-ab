import re
import pandas as pd
from datetime import datetime


# =====================================================
# smart_loader.py
# 支持三种 Excel 上传格式：
# 1. 单 sheet 用户级宽表
# 2. 双 sheet before / after
# 3. 多 sheet 标准结构：users / assignment / events / payments
# =====================================================


FIELD_ALIASES = {
    "user_id": [
        "user_id", "userid", "uid", "user", "username", "user_name",
        "用户id", "用户ID", "用户", "用户名", "账号", "账号id",
        "客户id", "客户", "会员id", "买家id", "openid", "手机号", "用户账号"
    ],

    "group": [
        "group", "ab_group", "variant", "version", "bucket",
        "实验组", "对照组", "分组", "组别", "实验分组", "版本", "新旧版本",
        "是否实验组", "ab组", "ab分组", "实验版本"
    ],

    "period": [
        "period", "stage", "phase", "before_after", "time_period",
        "阶段", "时期", "期间", "前后", "改版前后", "上线前后", "调整前后"
    ],

    "signup_date": [
        "signup_date", "register_time", "registration_date", "created_at",
        "注册时间", "注册日期", "创建时间", "开户时间"
    ],

    "region": [
        "region", "area", "city", "province", "location",
        "地区", "区域", "城市", "省份", "所在地"
    ],

    "device": [
        "device", "platform", "terminal", "os",
        "设备", "平台", "终端", "系统", "客户端"
    ],

    "user_type": [
        "user_type", "type", "customer_type", "segment",
        "用户类型", "客户类型", "人群", "用户分层", "用户标签"
    ],

    "assigned_at": [
        "assigned_at", "assign_time", "experiment_time",
        "分组时间", "实验时间", "进入实验时间"
    ],

    "event_time": [
        "event_time", "time", "timestamp", "behavior_time",
        "事件时间", "行为时间", "发生时间", "点击时间", "使用时间"
    ],

    "event_type": [
        "event_type", "event", "event_name", "action", "behavior", "behavior_type",
        "事件类型", "事件名称", "行为类型", "行为", "动作", "埋点事件"
    ],

    "payment_time": [
        "payment_time", "pay_time", "order_time", "purchase_time",
        "支付时间", "付费时间", "订单时间", "购买时间", "下单时间"
    ],

    "amount": [
        "amount", "total_amount", "revenue", "gmv", "sales", "fee", "price",
        "金额", "支付金额", "付费金额", "订单金额", "消费金额", "收入", "销售额",
        "客单价", "总金额", "总收入", "实付金额"
    ],

    "subscription_status": [
        "subscription_status", "pay_status", "status",
        "订阅状态", "支付状态", "付费状态", "订单状态"
    ],

    "login": [
        "login", "is_login", "logged_in",
        "登录", "是否登录", "登录行为", "活跃", "是否活跃"
    ],

    "view_feature": [
        "view_feature", "feature_view", "view", "exposure", "show",
        "曝光", "功能曝光", "新功能曝光", "浏览功能", "查看功能", "是否曝光"
    ],

    "click_feature": [
        "click_feature", "feature_click", "click", "clicked",
        "点击", "功能点击", "新功能点击", "是否点击", "点击新功能"
    ],

    "use_feature": [
        "use_feature", "feature_use", "used", "use", "usage",
        "使用", "功能使用", "新功能使用", "是否使用", "使用新功能"
    ],

    "complete_task": [
        "complete_task", "task_complete", "complete", "finished",
        "任务完成", "完成任务", "是否完成", "文档完成", "写作完成", "完成流程"
    ],

    "is_paid": [
        "is_paid", "paid", "converted", "payment_conversion",
        "是否付费", "付费", "是否转化", "转化", "付费转化", "是否购买", "购买"
    ],

    "payment_count": [
        "payment_count", "order_count", "orders", "num_orders",
        "订单数", "购买次数", "付费次数", "下单次数", "订单数量"
    ]
}


EVENT_FLAG_COLUMNS = [
    "login",
    "view_feature",
    "click_feature",
    "use_feature",
    "complete_task"
]


CONTROL_WORDS = {
    "control", "before", "pre", "old", "baseline", "a", "0",
    "对照组", "对照", "旧版本", "原版本", "改版前", "上线前", "调整前", "前", "之前"
}

TREATMENT_WORDS = {
    "treatment", "after", "post", "new", "b", "1",
    "实验组", "试验组", "新功能", "新版本", "改版后", "上线后", "调整后", "后", "之后"
}


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"[\s_\-（）()【】\[\]{}:：/\\|\.]+", "", x)
    return x


def build_alias_lookup():
    lookup = {}
    for standard_name, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            lookup[normalize_text(alias)] = standard_name
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def standardize_columns(df):
    """自动把中文 / 英文 / 近义字段名转成标准字段名。"""
    df = df.copy()
    rename_map = {}
    used_standard_names = set()

    for col in df.columns:
        normalized_col = normalize_text(col)
        if normalized_col in ALIAS_LOOKUP:
            standard_name = ALIAS_LOOKUP[normalized_col]
            if standard_name not in used_standard_names:
                rename_map[col] = standard_name
                used_standard_names.add(standard_name)

    df = df.rename(columns=rename_map)
    return df, rename_map


def normalize_group_values(series):
    def convert(x):
        x_raw = str(x).strip()
        x_norm = normalize_text(x_raw)

        for word in CONTROL_WORDS:
            if x_norm == normalize_text(word):
                return "control"

        for word in TREATMENT_WORDS:
            if x_norm == normalize_text(word):
                return "treatment"

        return x_raw

    return series.apply(convert)


def normalize_period_to_group(series):
    return normalize_group_values(series)


def to_binary(series):
    true_words = {"1", "true", "yes", "y", "是", "已", "有", "完成", "使用", "付费", "购买", "转化"}
    false_words = {"0", "false", "no", "n", "否", "无", "未", "没有", "未完成", "未使用", "未付费"}

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


def ensure_user_id(df, prefix="user"):
    """如果没有 user_id，就按行生成临时 user_id。"""
    df = df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = [f"{prefix}_{i + 1}" for i in range(len(df))]
        df["generated_user_id"] = True
    return df


def encode_user_ids(tables):
    """把字符串 user_id 编码成数字，兼容后续清洗逻辑。"""
    all_ids = []
    for df in tables.values():
        if df is not None and "user_id" in df.columns:
            all_ids.extend(df["user_id"].dropna().astype(str).tolist())

    unique_ids = list(pd.Series(all_ids).drop_duplicates())
    if not unique_ids:
        return tables

    numeric_ids = pd.to_numeric(pd.Series(unique_ids), errors="coerce")

    if numeric_ids.notna().all():
        for _, df in tables.items():
            if df is not None and "user_id" in df.columns:
                df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
        return tables

    id_map = {old_id: i + 1 for i, old_id in enumerate(unique_ids)}

    for _, df in tables.items():
        if df is not None and "user_id" in df.columns:
            df["original_user_id"] = df["user_id"]
            df["user_id"] = df["user_id"].astype(str).map(id_map)

    return tables


def classify_sheet_name(sheet_name):
    name = normalize_text(sheet_name)

    users_names = {"users", "user", "用户", "用户表"}
    assignment_names = {"assignment", "experimentassignment", "实验分组", "分组表", "assignment表"}
    events_names = {"events", "event", "行为", "行为表", "事件表", "埋点表"}
    payments_names = {"payments", "payment", "orders", "order", "付费", "订单", "订单表", "支付表"}

    if name in users_names:
        return "users"
    if name in assignment_names:
        return "assignment"
    if name in events_names:
        return "events"
    if name in payments_names:
        return "payments"

    for word in CONTROL_WORDS:
        word_norm = normalize_text(word)
        if name == word_norm or word_norm in name:
            return "before"

    for word in TREATMENT_WORDS:
        word_norm = normalize_text(word)
        if name == word_norm or word_norm in name:
            return "after"

    return "unknown"


def detect_table_type(df, sheet_name=""):
    sheet_type = classify_sheet_name(sheet_name)
    if sheet_type in ["users", "assignment", "events", "payments"]:
        return sheet_type

    cols = set(df.columns)
    has_user = "user_id" in cols
    has_group = "group" in cols
    has_period = "period" in cols
    has_event = "event_type" in cols or "event_time" in cols
    has_payment = "amount" in cols or "payment_time" in cols or "payment_count" in cols or "is_paid" in cols
    has_behavior_flags = any(col in cols for col in EVENT_FLAG_COLUMNS)
    has_user_profile = any(col in cols for col in ["signup_date", "region", "device", "user_type"])

    if (has_user or has_group or has_period) and (has_group or has_period) and (has_behavior_flags or has_payment):
        return "user_level"
    if has_user and has_event:
        return "events"
    if has_user and has_payment:
        return "payments"
    if has_user and has_group:
        return "assignment"
    if has_user and has_user_profile:
        return "users"
    if has_group and not has_user:
        return "summary"

    return "unknown"


def has_standard_structure(table_types):
    standard_types = {"users", "assignment", "events", "payments"}
    return len(standard_types.intersection(set(table_types))) >= 2


def find_before_after_pair(sheet_infos):
    before = None
    after = None

    for info in sheet_infos:
        sheet_role = classify_sheet_name(info["sheet_name"])
        if sheet_role == "before":
            before = info
        if sheet_role == "after":
            after = info

    if before is not None and after is not None:
        return before, after, "根据 sheet 名识别 before / after"

    # 兜底：只有两个 sheet，且不是标准多表，就按顺序识别。
    if len(sheet_infos) == 2:
        t1 = sheet_infos[0]["detected_type"]
        t2 = sheet_infos[1]["detected_type"]
        if t1 not in ["users", "assignment", "events", "payments"] and t2 not in ["users", "assignment", "events", "payments"]:
            return sheet_infos[0], sheet_infos[1], "未识别出明确名称，按 sheet 顺序识别：第一个 before，第二个 after"

    return None, None, ""


def convert_period_to_group_if_needed(df):
    df = df.copy()
    if "group" not in df.columns and "period" in df.columns:
        df["group"] = normalize_period_to_group(df["period"])
    if "group" in df.columns:
        df["group"] = normalize_group_values(df["group"])
    return df


def build_users_from_user_level(user_level):
    user_cols = ["user_id"]
    for col in ["signup_date", "region", "device", "user_type"]:
        if col in user_level.columns:
            user_cols.append(col)

    users = user_level[user_cols].drop_duplicates(subset=["user_id"]).copy()

    if "signup_date" not in users.columns:
        users["signup_date"] = pd.NaT
    if "region" not in users.columns:
        users["region"] = "未知地区"
    if "device" not in users.columns:
        users["device"] = "未知设备"
    if "user_type" not in users.columns:
        users["user_type"] = "unknown_user"

    return users


def build_assignment_from_user_level(user_level):
    assignment = user_level[["user_id", "group"]].drop_duplicates(subset=["user_id"]).copy()
    assignment["assigned_at"] = datetime(2026, 4, 1)
    return assignment


def build_events_from_user_level(user_level):
    rows = []
    default_time = datetime(2026, 4, 1)

    for _, row in user_level.iterrows():
        user_id = row["user_id"]
        for event_col in EVENT_FLAG_COLUMNS:
            if event_col in user_level.columns:
                try:
                    is_happened = float(row[event_col]) > 0
                except Exception:
                    is_happened = False

                if is_happened:
                    rows.append({
                        "user_id": user_id,
                        "event_time": default_time,
                        "event_type": event_col
                    })

    return pd.DataFrame(rows, columns=["user_id", "event_time", "event_type"])


def build_payments_from_user_level(user_level):
    rows = []
    default_time = datetime(2026, 4, 1)

    for _, row in user_level.iterrows():
        user_id = row["user_id"]
        amount = 0

        if "amount" in user_level.columns and not pd.isna(row.get("amount")):
            try:
                amount = float(row.get("amount"))
            except Exception:
                amount = 0

        is_paid = 0
        if "is_paid" in user_level.columns:
            is_paid = int(to_binary(pd.Series([row.get("is_paid")])).iloc[0])
        elif "payment_count" in user_level.columns:
            try:
                is_paid = 1 if float(row.get("payment_count")) > 0 else 0
            except Exception:
                is_paid = 0
        elif amount > 0:
            is_paid = 1

        if is_paid == 1:
            final_amount = amount if amount > 0 else 1
            rows.append({
                "user_id": user_id,
                "payment_time": default_time,
                "amount": final_amount,
                "subscription_status": "paid"
            })

    return pd.DataFrame(rows, columns=["user_id", "payment_time", "amount", "subscription_status"])


def convert_user_level_to_standard_tables(user_level):
    user_level = user_level.copy()
    user_level = ensure_user_id(user_level, prefix="row")
    user_level = convert_period_to_group_if_needed(user_level)

    if "group" not in user_level.columns:
        raise ValueError(
            "用户级宽表缺少分组字段。请提供 group / 实验组 / 分组 / 版本，"
            "或 period / 阶段 / 改版前后。"
        )

    valid_groups = {"control", "treatment"}
    if not set(user_level["group"].dropna().astype(str)).intersection(valid_groups):
        raise ValueError(
            "无法识别 control / treatment。请把分组写成 control/treatment，"
            "或 before/after，或 改版前/改版后。"
        )

    users = build_users_from_user_level(user_level)
    assignment = build_assignment_from_user_level(user_level)
    events = build_events_from_user_level(user_level)
    payments = build_payments_from_user_level(user_level)

    return users, assignment, events, payments


def combine_before_after_sheets(before_df, after_df):
    """
    双 sheet before / after 合并。
    统一生成新的 user_id，避免同一个用户在前后两期重复导致 assignment 去重丢数据。
    """
    before_df = before_df.copy()
    after_df = after_df.copy()

    if "user_id" in before_df.columns:
        before_df["original_user_id"] = before_df["user_id"]
    if "user_id" in after_df.columns:
        after_df["original_user_id"] = after_df["user_id"]

    before_df["user_id"] = [f"before_{i + 1}" for i in range(len(before_df))]
    after_df["user_id"] = [f"after_{i + 1}" for i in range(len(after_df))]

    before_df["group"] = "control"
    after_df["group"] = "treatment"

    return pd.concat([before_df, after_df], ignore_index=True)


def build_users_from_available_data(assignment, events, payments):
    user_ids = []
    for df in [assignment, events, payments]:
        if df is not None and "user_id" in df.columns:
            user_ids.extend(df["user_id"].dropna().tolist())

    user_ids = pd.Series(user_ids).drop_duplicates()

    return pd.DataFrame({
        "user_id": user_ids,
        "signup_date": pd.NaT,
        "region": "未知地区",
        "device": "未知设备",
        "user_type": "unknown_user"
    })


def finalize_standard_tables(users, assignment, events, payments):
    if users is None:
        users = build_users_from_available_data(assignment, events, payments)

    if assignment is None:
        for df in [users, events, payments]:
            if df is not None and "user_id" in df.columns and "group" in df.columns:
                assignment = df[["user_id", "group"]].drop_duplicates()
                assignment["assigned_at"] = datetime(2026, 4, 1)
                break

    if assignment is None:
        raise ValueError(
            "无法识别实验分组字段。请提供 group / 实验组 / 分组 / 版本，"
            "或使用 before / after 双 sheet 格式。"
        )

    if "group" in assignment.columns:
        assignment["group"] = normalize_group_values(assignment["group"])

    if events is None:
        events = pd.DataFrame(columns=["user_id", "event_time", "event_type"])

    if payments is None:
        payments = pd.DataFrame(columns=["user_id", "payment_time", "amount", "subscription_status"])

    tables = encode_user_ids({
        "users": users,
        "assignment": assignment,
        "events": events,
        "payments": payments
    })

    users = tables["users"]
    assignment = tables["assignment"]
    events = tables["events"]
    payments = tables["payments"]

    if "user_id" not in users.columns:
        users = build_users_from_available_data(assignment, events, payments)

    for col in ["signup_date", "region", "device", "user_type"]:
        if col not in users.columns:
            if col == "signup_date":
                users[col] = pd.NaT
            elif col == "region":
                users[col] = "未知地区"
            elif col == "device":
                users[col] = "未知设备"
            elif col == "user_type":
                users[col] = "unknown_user"

    if "assigned_at" not in assignment.columns:
        assignment["assigned_at"] = datetime(2026, 4, 1)

    if "event_time" not in events.columns:
        events["event_time"] = datetime(2026, 4, 1)
    if "event_type" not in events.columns:
        events["event_type"] = ""

    if "payment_time" not in payments.columns:
        payments["payment_time"] = datetime(2026, 4, 1)
    if "amount" not in payments.columns:
        payments["amount"] = 0
    if "subscription_status" not in payments.columns:
        payments["subscription_status"] = "paid"

    return users, assignment, events, payments


def load_experiment_excel(uploaded_file):
    """
    智能读取 Excel。

    支持三种格式：
    1. 单 sheet 用户级宽表
    2. 双 sheet before / after
    3. 多 sheet 标准结构：users / assignment / events / payments
    """
    excel = pd.ExcelFile(uploaded_file)

    sheet_infos = []
    report_rows = []

    for sheet_name in excel.sheet_names:
        raw_df = excel.parse(sheet_name)
        if raw_df.empty:
            continue

        df, rename_map = standardize_columns(raw_df)
        df = convert_period_to_group_if_needed(df)

        for col in EVENT_FLAG_COLUMNS + ["is_paid"]:
            if col in df.columns:
                df[col] = to_binary(df[col])

        table_type = detect_table_type(df, sheet_name)

        sheet_infos.append({
            "sheet_name": sheet_name,
            "df": df,
            "detected_type": table_type,
            "rename_map": rename_map
        })

        report_rows.append({
            "sheet_name": sheet_name,
            "detected_type": table_type,
            "rows": len(df),
            "mapped_columns": str(rename_map),
            "note": ""
        })

    if not sheet_infos:
        raise ValueError("上传的 Excel 中没有可读取的数据。")

    detect_report = pd.DataFrame(report_rows)

    # 模式 1：单 sheet 用户级宽表
    if len(sheet_infos) == 1:
        info = sheet_infos[0]
        users, assignment, events, payments = convert_user_level_to_standard_tables(info["df"])
        detect_report.loc[0, "note"] = "单 sheet 用户级宽表，已转换为标准实验分析数据。"
        users, assignment, events, payments = finalize_standard_tables(users, assignment, events, payments)
        return users, assignment, events, payments, detect_report

    # 模式 2：双 sheet before / after
    before_info, after_info, before_after_note = find_before_after_pair(sheet_infos)
    table_types = [x["detected_type"] for x in sheet_infos]

    if before_info is not None and after_info is not None and not has_standard_structure(table_types):
        user_level = combine_before_after_sheets(before_info["df"], after_info["df"])
        users, assignment, events, payments = convert_user_level_to_standard_tables(user_level)
        detect_report.loc[
            detect_report["sheet_name"].isin([before_info["sheet_name"], after_info["sheet_name"]]),
            "note"
        ] = before_after_note + "，已转换为 control / treatment。"
        users, assignment, events, payments = finalize_standard_tables(users, assignment, events, payments)
        return users, assignment, events, payments, detect_report

    # 模式 3：多 sheet 标准结构
    detected_tables = {
        "users": None,
        "assignment": None,
        "events": None,
        "payments": None,
        "user_level": None
    }

    for info in sheet_infos:
        table_type = info["detected_type"]
        if table_type in detected_tables and detected_tables[table_type] is None:
            detected_tables[table_type] = info["df"]

    if detected_tables["user_level"] is not None:
        users, assignment, events, payments = convert_user_level_to_standard_tables(detected_tables["user_level"])
        detect_report.loc[detect_report["detected_type"] == "user_level", "note"] = "识别为用户级宽表，已转换为标准实验分析数据。"
    else:
        users = detected_tables["users"]
        assignment = detected_tables["assignment"]
        events = detected_tables["events"]
        payments = detected_tables["payments"]

    users, assignment, events, payments = finalize_standard_tables(users, assignment, events, payments)
    return users, assignment, events, payments, detect_report

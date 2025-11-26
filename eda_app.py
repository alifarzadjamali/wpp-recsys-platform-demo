import os
import duckdb
import polars as pl
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

DATA_PATH_DEFAULT = "wpp_2021_2023_final.parquet"
OUTPUT_DIR = "eda_outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figs")


# ============================================
# HELPERS
# ============================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def save_figure(fig, filename, export_figs):
    if not export_figs:
        return
    ensure_output_dir()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches="tight")


def add_bar_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        if height is None or pd.isna(height) or height == 0:
            continue
        ax.annotate(
            f"{height:.0f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def pl_to_pd(df_pl: pl.DataFrame) -> pd.DataFrame:
    if df_pl.is_empty():
        return pd.DataFrame()
    return df_pl.to_pandas()


# ============================================
# DUCKDB CONNECTION + METADATA
# ============================================

@st.cache_resource
def connect_duckdb():
    con = duckdb.connect()
    return con


@st.cache_data
def get_available_years(data_path: str) -> list:
    con = connect_duckdb()
    q = f"""
        SELECT DISTINCT year(createdAt) AS y
        FROM read_parquet('{data_path}')
        ORDER BY y
    """
    years = [int(r[0]) for r in con.execute(q).fetchall()]
    return years


@st.cache_data
def get_columns(data_path: str) -> list:
    con = connect_duckdb()
    q = f"SELECT * FROM read_parquet('{data_path}') LIMIT 0"
    df = con.execute(q).fetchnumpy()
    return list(df.keys())


def has_column(data_path: str, col_name: str) -> bool:
    cols = get_columns(data_path)
    return col_name in cols


# ============================================
# GENERIC QUERY HELPER
# ============================================

@st.cache_data
def run_query(data_path: str, sql: str) -> pl.DataFrame:
    con = connect_duckdb()
    return pl.from_arrow(con.execute(sql).fetch_arrow_table())


# ============================================
# BASIC SAMPLE
# ============================================

def show_basic_sample(data_path: str):
    st.subheader("Basic sample")

    q = f"SELECT * FROM read_parquet('{data_path}') LIMIT 5"
    df_pl = run_query(data_path, q)
    st.write(pl_to_pd(df_pl))

    con = connect_duckdb()
    con.execute(f"""
        CREATE OR REPLACE VIEW tmp_parquet AS 
        SELECT * FROM read_parquet('{data_path}') LIMIT 0
    """)
    info = con.execute("DESCRIBE tmp_parquet").fetchdf()

    st.text("Columns and types:")
    st.dataframe(info)


# ============================================
# UNIQUE COUNTS BY YEAR (PATCHED)
# ============================================

def unique_counts_by_year(data_path: str, active_years, anon_col: str):
    st.subheader("Unique counts by year")

    years_tuple = tuple(active_years)

    # per year counts (members, anonymous, merchants, partners)
    q_counts = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                member,
                {anon_col} AS anon_id,
                merchant,
                partner,
                categories
            FROM read_parquet('{data_path}')
        )
        SELECT
            y AS year,
            COUNT(DISTINCT IF(NOT is_anonymous, member, NULL)) AS unique_members,
            COUNT(DISTINCT IF(is_anonymous, anon_id, NULL)) AS unique_anonymous,
            COUNT(DISTINCT merchant) AS unique_merchants,
            COUNT(DISTINCT partner) AS unique_partners
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY y
        ORDER BY y
    """
    per_year = run_query(data_path, q_counts)

    # per year unique categories
    q_cats = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        )
        SELECT
            y AS year,
            COUNT(DISTINCT category_id) AS unique_categories
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY y
        ORDER BY y
    """
    cats_year = run_query(data_path, q_cats)

    # join per year
    per_year_pd = pl_to_pd(per_year)
    cats_year_pd = pl_to_pd(cats_year)
    df_year = per_year_pd.merge(cats_year_pd, on="year", how="left")

    # overall row
    q_overall = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                member,
                {anon_col} AS anon_id,
                merchant,
                partner,
                categories
            FROM read_parquet('{data_path}')
        )
        SELECT
            COUNT(DISTINCT IF(NOT is_anonymous, member, NULL)) AS unique_members,
            COUNT(DISTINCT IF(is_anonymous, anon_id, NULL)) AS unique_anonymous,
            COUNT(DISTINCT merchant) AS unique_merchants,
            COUNT(DISTINCT partner) AS unique_partners
        FROM base
        WHERE y IN {years_tuple}
    """
    overall = run_query(data_path, q_overall)
    overall_pd = pl_to_pd(overall)
    overall_pd["year"] = "ALL"

    # overall unique categories
    q_overall_cat = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        )
        SELECT COUNT(DISTINCT category_id) AS unique_categories
        FROM base
        WHERE y IN {years_tuple}
    """
    overall_cat = pl_to_pd(run_query(data_path, q_overall_cat))
    overall_pd["unique_categories"] = overall_cat["unique_categories"].iloc[0]

    result = pd.concat([df_year, overall_pd], ignore_index=True)
    ensure_output_dir()
    result.to_csv(os.path.join(OUTPUT_DIR, "unique_counts_by_year.csv"), index=False)
    st.dataframe(result)


# ============================================
# TOP ENTITIES IN FOCUS PERIOD
# ============================================

def top3_focus_entities(data_path: str, active_years, start_ym: str, end_ym: str):
    years_tuple = tuple(active_years)
    start_date = f"{start_ym}-01"
    end_year, end_month = map(int, end_ym.split("-"))
    if end_month == 12:
        end_filter = f"{end_year + 1}-01-01"
    else:
        end_filter = f"{end_year}-{end_month + 1:02d}-01"

    q_partners = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                member,
                partner
            FROM read_parquet('{data_path}')
        )
        SELECT
            partner,
            COUNT(DISTINCT member) AS unique_members,
            COUNT(*) AS member_views
        FROM base
        WHERE y IN {years_tuple}
          AND createdAt >= '{start_date}'
          AND createdAt <  '{end_filter}'
          AND NOT is_anonymous
          AND partner != 0
        GROUP BY partner
        ORDER BY unique_members DESC
        LIMIT 3
    """

    q_merchants = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                merchant,
                name
            FROM read_parquet('{data_path}')
        )
        SELECT
            merchant,
            name,
            COUNT(*) AS view_count
        FROM base
        WHERE y IN {years_tuple}
          AND createdAt >= '{start_date}'
          AND createdAt <  '{end_filter}'
        GROUP BY merchant, name
        ORDER BY view_count DESC
        LIMIT 3
    """

    q_categories = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        )
        SELECT
            category_id,
            COUNT(*) AS view_count
        FROM base
        WHERE y IN {years_tuple}
          AND createdAt >= '{start_date}'
          AND createdAt <  '{end_filter}'
        GROUP BY category_id
        ORDER BY view_count DESC
        LIMIT 3
    """

    p = run_query(data_path, q_partners)
    m = run_query(data_path, q_merchants)
    c = run_query(data_path, q_categories)
    return pl_to_pd(p), pl_to_pd(m), pl_to_pd(c)


def show_top3_focus_bars(data_path: str, active_years, start_ym, end_ym, export_figs: bool):
    st.subheader("Top 3 entities in focus period")

    partners, merchants, categories = top3_focus_entities(data_path, active_years, start_ym, end_ym)

    col1, col2, col3 = st.columns(3)

    if not partners.empty:
        with col1:
            st.markdown("**Top 3 partners (by unique members)**")
            st.dataframe(partners)
            fig, ax = plt.subplots()
            sns.barplot(data=partners, x="partner", y="unique_members", ax=ax)
            add_bar_labels(ax, ax.patches)
            ax.set_xlabel("Partner")
            ax.set_ylabel("Unique members")
            fig.tight_layout()
            save_figure(fig, "top3_partners_focus.jpg", export_figs)
            st.pyplot(fig)
            plt.close(fig)

    if not merchants.empty:
        with col2:
            st.markdown("**Top 3 merchants (by views)**")
            st.dataframe(merchants)
            fig, ax = plt.subplots()
            sns.barplot(data=merchants, x="name", y="view_count", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            add_bar_labels(ax, ax.patches)
            ax.set_xlabel("Merchant")
            ax.set_ylabel("Views")
            fig.tight_layout()
            save_figure(fig, "top3_merchants_focus.jpg", export_figs)
            st.pyplot(fig)
            plt.close(fig)

    if not categories.empty:
        with col3:
            st.markdown("**Top 3 categories (by views)**")
            st.dataframe(categories)
            fig, ax = plt.subplots()
            sns.barplot(data=categories, x="category_id", y="view_count", ax=ax)
            add_bar_labels(ax, ax.patches)
            ax.set_xlabel("Category ID")
            ax.set_ylabel("Views")
            fig.tight_layout()
            save_figure(fig, "top3_categories_focus.jpg", export_figs)
            st.pyplot(fig)
            plt.close(fig)


# ============================================
# MEMBERS PER YEAR
# ============================================

def members_per_year(data_path: str, active_years, anon_col: str, export_figs: bool):
    st.subheader("Members vs anonymous per year")
    years_tuple = tuple(active_years)

    q = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                member,
                {anon_col} AS anon_id
            FROM read_parquet('{data_path}')
        )
        SELECT
            y AS year,
            COUNT(DISTINCT IF(NOT is_anonymous, member, NULL)) AS members,
            COUNT(DISTINCT IF(is_anonymous, anon_id, NULL)) AS anonymous
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY y
        ORDER BY y
    """
    df = pl_to_pd(run_query(data_path, q))

    x = range(len(df))
    fig, ax = plt.subplots()
    rects1 = ax.bar([i - 0.2 for i in x], df["members"], width=0.4, label="Members")
    rects2 = ax.bar([i + 0.2 for i in x], df["anonymous"], width=0.4, label="Anonymous")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["year"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique users")
    ax.legend()
    add_bar_labels(ax, rects1)
    add_bar_labels(ax, rects2)

    fig.tight_layout()
    save_figure(fig, "members_per_year.jpg", export_figs)
    st.pyplot(fig)
    plt.close(fig)


# ============================================
# CATEGORY PIES AND FOCUS COMPARISON
# ============================================

def category_pies_and_focus(data_path: str, active_years, start_ym: str, end_ym: str, export_figs: bool):
    st.subheader("Category share and focus comparison")
    years_tuple = tuple(active_years)

    q_overall = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        )
        SELECT
            category_id,
            COUNT(*) AS views
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY category_id
        ORDER BY category_id
    """
    overall = pl_to_pd(run_query(data_path, q_overall))

    if not overall.empty:
        fig, ax = plt.subplots()
        ax.pie(overall["views"], labels=overall["category_id"], autopct="%1.1f%%")
        ax.set_title("Category share of views (active years)")
        fig.tight_layout()
        save_figure(fig, "category_share_active_years.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

    for y in active_years:
        q_year = f"""
            WITH base AS (
                SELECT
                    year(createdAt) AS y,
                    UNNEST(categories) AS category_id
                FROM read_parquet('{data_path}')
            )
            SELECT
                category_id,
                COUNT(*) AS views
            FROM base
            WHERE y = {y}
            GROUP BY category_id
            ORDER BY category_id
        """
        df_y = pl_to_pd(run_query(data_path, q_year))
        if df_y.empty:
            continue
        fig, ax = plt.subplots()
        ax.pie(df_y["views"], labels=df_y["category_id"], autopct="%1.1f%%")
        ax.set_title(f"Category share of views in {y}")
        fig.tight_layout()
        save_figure(fig, f"category_share_{y}.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

    start_date = f"{start_ym}-01"
    end_year, end_month = map(int, end_ym.split("-"))
    if end_month == 12:
        end_filter = f"{end_year + 1}-01-01"
    else:
        end_filter = f"{end_year}-{end_month + 1:02d}-01"

    q_focus = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        )
        SELECT
            category_id,
            is_anonymous,
            COUNT(*) AS views
        FROM base
        WHERE y IN {years_tuple}
          AND createdAt >= '{start_date}'
          AND createdAt <  '{end_filter}'
        GROUP BY category_id, is_anonymous
    """
    df_focus = pl_to_pd(run_query(data_path, q_focus))
    if df_focus.empty:
        st.info("No focus data for category comparison.")
        return

    members_df = df_focus[df_focus["is_anonymous"] == False].copy()
    anon_df = df_focus[df_focus["is_anonymous"] == True].copy()

    all_cats = sorted(set(df_focus["category_id"]))
    m_vals = [members_df.loc[members_df["category_id"] == c, "views"].sum() for c in all_cats]
    a_vals = [anon_df.loc[anon_df["category_id"] == c, "views"].sum() for c in all_cats]

    y_idx = range(len(all_cats))
    fig, ax = plt.subplots(figsize=(8, max(4, len(all_cats) * 0.3)))
    ax.barh(y_idx, m_vals, label="Members")
    ax.barh(y_idx, a_vals, left=m_vals, label="Anonymous")
    ax.set_yticks(list(y_idx))
    ax.set_yticklabels(all_cats)
    ax.set_xlabel("Views")
    ax.set_title("Category views in focus period (members vs anonymous)")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "category_members_vs_anon_focus.jpg", export_figs)
    st.pyplot(fig)
    plt.close(fig)


# ============================================
# TOP PARTNERS BY MEMBERS (FOCUS)
# ============================================

def top_partners_focus(data_path: str, active_years, start_ym, end_ym, top_k: int, export_figs: bool):
    st.subheader("Top partners by members (focus period)")
    years_tuple = tuple(active_years)
    start_date = f"{start_ym}-01"
    end_year, end_month = map(int, end_ym.split("-"))
    if end_month == 12:
        end_filter = f"{end_year + 1}-01-01"
    else:
        end_filter = f"{end_year}-{end_month + 1:02d}-01"

    q = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                is_anonymous,
                member,
                partner
            FROM read_parquet('{data_path}')
        )
        SELECT
            partner,
            COUNT(DISTINCT member) AS unique_members,
            COUNT(*) AS member_views
        FROM base
        WHERE y IN {years_tuple}
          AND createdAt >= '{start_date}'
          AND createdAt <  '{end_filter}'
          AND NOT is_anonymous
          AND partner != 0
        GROUP BY partner
        ORDER BY unique_members DESC
        LIMIT {top_k}
    """
    df = pl_to_pd(run_query(data_path, q))
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, "top_partners_by_members_focus.csv"), index=False)

    if df.empty:
        st.info("No data for partners in focus period.")
        return

    x = range(len(df))
    fig, ax = plt.subplots()
    rects1 = ax.bar([i - 0.2 for i in x], df["unique_members"], width=0.4, label="Unique members")
    rects2 = ax.bar([i + 0.2 for i in x], df["member_views"], width=0.4, label="Member views")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["partner"], rotation=45, ha="right")
    ax.legend()
    add_bar_labels(ax, rects1)
    add_bar_labels(ax, rects2)

    fig.tight_layout()
    save_figure(fig, "top_partners_by_members_focus.jpg", export_figs)
    st.pyplot(fig)
    plt.close(fig)


# ============================================
# PAID LEVEL ANALYSIS
# ============================================

def paid_level_analysis(data_path: str, active_years, export_figs: bool):
    st.subheader("Views by paid level (active years)")
    years_tuple = tuple(active_years)

    q_total = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                level
            FROM read_parquet('{data_path}')
        )
        SELECT
            level,
            COUNT(*) AS total_views
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY level
        ORDER BY level
    """
    total = pl_to_pd(run_query(data_path, q_total))

    q_avg = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                level,
                merchant,
                date_trunc('month', createdAt) AS ym
            FROM read_parquet('{data_path}')
        ),
        per_month AS (
            SELECT
                level,
                merchant,
                ym,
                COUNT(*) AS views
            FROM base
            WHERE y IN {years_tuple}
            GROUP BY level, merchant, ym
        ),
        merchant_avg AS (
            SELECT
                level,
                merchant,
                AVG(views) AS avg_views
            FROM per_month
            GROUP BY level, merchant
        )
        SELECT
            level,
            AVG(avg_views) AS avg_monthly_per_merchant
        FROM merchant_avg
        GROUP BY level
        ORDER BY level
    """
    avg = pl_to_pd(run_query(data_path, q_avg))

    df = total.merge(avg, on="level", how="left")
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, "merchant_views_by_paid_level.csv"), index=False)
    st.dataframe(df)

    if not df.empty:
        fig, ax = plt.subplots()
        rects1 = ax.bar(df["level"].astype(str), df["total_views"])
        ax.set_xlabel("Level")
        ax.set_ylabel("Total views")
        add_bar_labels(ax, rects1)
        fig.tight_layout()
        save_figure(fig, "views_by_paid_level_total.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        rects2 = ax.bar(df["level"].astype(str), df["avg_monthly_per_merchant"])
        ax.set_xlabel("Level")
        ax.set_ylabel("Avg monthly views / merchant")
        add_bar_labels(ax, rects2)
        fig.tight_layout()
        save_figure(fig, "views_by_paid_level_avg_month.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)


# ============================================
# TOP 3 MERCHANTS PER CATEGORY (FOCUS)
# ============================================

def top3_merchants_per_category(data_path: str, active_years, start_ym, end_ym):
    st.subheader("Top 3 merchants per category (focus period)")
    years_tuple = tuple(active_years)
    start_date = f"{start_ym}-01"
    end_year, end_month = map(int, end_ym.split("-"))
    if end_month == 12:
        end_filter = f"{end_year + 1}-01-01"
    else:
        end_filter = f"{end_year}-{end_month + 1:02d}-01"

    q = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                merchant,
                name,
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
        ),
        cat AS (
            SELECT
                category_id,
                merchant,
                name,
                COUNT(*) AS views
            FROM base
            WHERE y IN {years_tuple}
              AND createdAt >= '{start_date}'
              AND createdAt <  '{end_filter}'
            GROUP BY category_id, merchant, name
        )
        SELECT *
        FROM (
            SELECT
                category_id,
                merchant,
                name,
                views,
                ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY views DESC) AS rn
            FROM cat
        )
        WHERE rn <= 3
        ORDER BY category_id, views DESC
    """
    df = pl_to_pd(run_query(data_path, q))
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, "top3_merchants_per_category_focus.csv"), index=False)
    st.dataframe(df)


# ============================================
# TIME BASED TRENDS
# ============================================

def time_trends(data_path: str, active_years, export_figs: bool):
    st.subheader("Time based trends (monthly and hourly)")
    years_tuple = tuple(active_years)

    q_monthly_all = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                date_trunc('month', createdAt) AS ym
            FROM read_parquet('{data_path}')
        )
        SELECT
            ym,
            COUNT(*) AS views
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY ym
        ORDER BY ym
    """
    monthly_all = pl_to_pd(run_query(data_path, q_monthly_all))
    if not monthly_all.empty:
        monthly_all["label"] = pd.to_datetime(monthly_all["ym"]).dt.strftime("%y-%-m")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(monthly_all)), monthly_all["views"], marker="o")
        ax.set_xticks(range(len(monthly_all)))
        ax.set_xticklabels(monthly_all["label"], rotation=45, ha="right")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Total views")
        fig.tight_layout()
        save_figure(fig, "monthly_total_views_active_years.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

    q_monthly = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                month(createdAt) AS m
            FROM read_parquet('{data_path}')
        )
        SELECT
            y AS year,
            m AS month,
            COUNT(*) AS views
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY year, month
        ORDER BY year, month
    """
    monthly = pl_to_pd(run_query(data_path, q_monthly))
    if not monthly.empty:
        fig, ax = plt.subplots()
        for y in sorted(monthly["year"].unique()):
            g = monthly[monthly["year"] == y]
            ax.plot(g["month"], g["views"], marker="o", label=str(y))
        ax.set_xlabel("Month")
        ax.set_ylabel("Total views")
        ax.set_xticks(range(1, 13))
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "monthly_views_comparison_years.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

    last_year = max(active_years)

    q_monthly_avg = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                CAST(createdAt AS DATE) AS d
            FROM read_parquet('{data_path}')
        ),
        per_day AS (
            SELECT
                d,
                COUNT(*) AS views
            FROM base
            WHERE y = {last_year}
            GROUP BY d
        )
        SELECT
            EXTRACT('month' FROM d) AS month,
            AVG(views) AS avg_daily_views
        FROM per_day
        GROUP BY month
        ORDER BY month
    """
    monthly_avg = pl_to_pd(run_query(data_path, q_monthly_avg))
    if not monthly_avg.empty:
        fig, ax = plt.subplots()
        ax.plot(monthly_avg["month"], monthly_avg["avg_daily_views"], marker="o")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average daily views")
        ax.set_xticks(range(1, 13))
        fig.tight_layout()
        save_figure(fig, f"monthly_avg_daily_views_{last_year}.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)

    q_hourly_counts = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                CAST(createdAt AS DATE) AS d,
                EXTRACT('hour' FROM createdAt) AS h
            FROM read_parquet('{data_path}')
        ),
        filtered AS (
            SELECT * FROM base WHERE y = {last_year}
        ),
        per_hour AS (
            SELECT
                h AS hour,
                COUNT(*) AS views
            FROM filtered
            GROUP BY hour
        ),
        days AS (
            SELECT COUNT(DISTINCT d) AS num_days FROM filtered
        )
        SELECT
            per_hour.hour,
            CAST(per_hour.views AS DOUBLE) / days.num_days AS avg_views
        FROM per_hour
        CROSS JOIN days
        ORDER BY per_hour.hour
    """
    hourly = pl_to_pd(run_query(data_path, q_hourly_counts))
    if not hourly.empty:
        full_hours = pd.DataFrame({"hour": range(24)})
        hourly = full_hours.merge(hourly, on="hour", how="left").fillna(0)
        fig, ax = plt.subplots()
        sns.lineplot(data=hourly, x="hour", y="avg_views", marker="o", ax=ax)
        ax.set_xlabel("Hour of day (0–23)")
        ax.set_ylabel("Average views per day in that hour")
        ax.set_xticks(range(24))
        fig.tight_layout()
        save_figure(fig, f"avg_hourly_views_{last_year}.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)


# ============================================
# SESSION REPORTS
# ============================================

def session_reports(data_path: str, active_years, export_figs: bool):
    st.subheader("Session reports")
    if not has_column(data_path, "session_id"):
        st.info("No session_id column in data.")
        return

    years_tuple = tuple(active_years)
    q = f"""
        WITH base AS (
            SELECT
                createdAt,
                year(createdAt) AS y,
                session_id,
                partner,
                member,
                session_seq,
                merchant
            FROM read_parquet('{data_path}')
        )
        SELECT
            session_id,
            FIRST(partner) AS partner,
            FIRST(member) AS member,
            MIN(createdAt) AS start,
            MAX(createdAt) AS end,
            MAX(session_seq) AS events,
            COUNT(DISTINCT merchant) AS merchants,
            DATEDIFF('second', MIN(createdAt), MAX(createdAt)) AS duration_sec
        FROM base
        WHERE y IN {years_tuple}
        GROUP BY session_id
        ORDER BY session_id
    """
    df = pl_to_pd(run_query(data_path, q))
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, "session_summary_active_years.csv"), index=False)

    st.dataframe(df.head(50))

    if not df.empty:
        fig, ax = plt.subplots()
        ax.hist(df["events"], bins=20)
        ax.set_xlabel("Events in session")
        ax.set_ylabel("Count")
        fig.tight_layout()
        save_figure(fig, "session_length_distribution.jpg", export_figs)
        st.pyplot(fig)
        plt.close(fig)


# ============================================
# DRILLDOWNS
# ============================================

def partner_drilldown(data_path: str, partner_id: int, top_k_cat: int):
    st.subheader(f"Partner {partner_id} drilldown")

    q_summary = f"""
        SELECT
            COUNT(*) AS total_views,
            COUNT(DISTINCT member) FILTER (WHERE NOT is_anonymous) AS unique_members
        FROM read_parquet('{data_path}')
        WHERE partner = {partner_id}
    """
    summary = pl_to_pd(run_query(data_path, q_summary))
    if summary.empty or summary["total_views"].iloc[0] == 0:
        st.warning("No data for this partner.")
        return

    total_views = int(summary["total_views"].iloc[0])
    unique_members = int(summary["unique_members"].iloc[0])
    st.write(f"Total views: **{total_views}**")
    st.write(f"Unique members: **{unique_members}**")

    q_levels = f"""
        SELECT
            level,
            COUNT(*) AS count
        FROM read_parquet('{data_path}')
        WHERE partner = {partner_id}
        GROUP BY level
        ORDER BY level
    """
    lvl = pl_to_pd(run_query(data_path, q_levels))
    st.write("Paid level distribution:")
    st.dataframe(lvl)

    q_cat = f"""
        SELECT
            category_id,
            COUNT(*) AS views
        FROM (
            SELECT
                UNNEST(categories) AS category_id
            FROM read_parquet('{data_path}')
            WHERE partner = {partner_id}
        )
        GROUP BY category_id
        ORDER BY views DESC
        LIMIT {top_k_cat}
    """
    cats = pl_to_pd(run_query(data_path, q_cat))
    st.write(f"Top {top_k_cat} categories for this partner:")
    st.dataframe(cats)


def category_drilldown(data_path: str, category_id: int, top_k_merchants: int, active_years):
    st.subheader(f"Category {category_id} drilldown")
    years_tuple = tuple(active_years)

    q_cat_counts = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                merchant,
                UNNEST(categories) AS cat
            FROM read_parquet('{data_path}')
        )
        SELECT
            COUNT(*) AS views,
            COUNT(DISTINCT merchant) AS merchants
        FROM base
        WHERE y IN {years_tuple}
          AND cat = {category_id}
    """
    cat_counts = pl_to_pd(run_query(data_path, q_cat_counts))
    if cat_counts.empty or cat_counts["views"].iloc[0] == 0:
        st.warning("No data for this category.")
        return

    views_cat = int(cat_counts["views"].iloc[0])
    merchants_cat = int(cat_counts["merchants"].iloc[0])

    q_total_cat = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                UNNEST(categories) AS cat
            FROM read_parquet('{data_path}')
        )
        SELECT COUNT(*) AS total_cat_views
        FROM base
        WHERE y IN {years_tuple}
    """

    total_cat = pl_to_pd(run_query(data_path, q_total_cat))

    # ---- FIX ----
    if total_cat.empty or "total_cat_views" not in total_cat.columns:
        st.warning("No overall category view data for these years.")
        return
    # --------------

    total_cat_views = int(total_cat["total_cat_views"].iloc[0])
    share = views_cat / total_cat_views if total_cat_views > 0 else 0.0


    st.write(f"Views in category: **{views_cat}**")
    st.write(f"Share of all category views: **{share:.2%}**")
    st.write(f"Number of merchants: **{merchants_cat}**")

    q_top_merchants = f"""
        WITH base AS (
            SELECT
                year(createdAt) AS y,
                merchant,
                name,
                UNNEST(categories) AS cat
            FROM read_parquet('{data_path}')
        )
        SELECT
            merchant,
            name,
            COUNT(*) AS views_in_category
        FROM base
        WHERE y IN {years_tuple}
          AND cat = {category_id}
        GROUP BY merchant, name
        ORDER BY views_in_category DESC
        LIMIT {top_k_merchants}
    """
    top_m = pl_to_pd(run_query(data_path, q_top_merchants))

    if not top_m.empty:
        merchants_list = tuple(top_m["merchant"].tolist())
        q_levels = f"""
            SELECT
                merchant,
                level,
                COUNT(*) AS cnt
            FROM read_parquet('{data_path}')
            WHERE merchant IN {merchants_list}
            GROUP BY merchant, level
        """
        lv = pl_to_pd(run_query(data_path, q_levels))
        lv = lv.sort_values(["merchant", "cnt"], ascending=[True, False]).drop_duplicates("merchant")
        lv = lv[["merchant", "level"]].rename(columns={"level": "paid_level"})

        merged = top_m.merge(lv, on="merchant", how="left")
        merged["share_of_cat"] = merged["views_in_category"] / views_cat
    else:
        merged = top_m
        merged["paid_level"] = None
        merged["share_of_cat"] = 0.0

    ensure_output_dir()
    merged.to_csv(os.path.join(OUTPUT_DIR, f"category_{category_id}_top_merchants.csv"), index=False)
    st.write("Top merchants in this category:")
    st.dataframe(merged)


# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.set_page_config(page_title="WPP EDA (DuckDB + Polars)", layout="wide")
    st.title("WPP Merchant Views – DuckDB + Polars EDA Dashboard")

    st.sidebar.header("Data & configuration")

    data_path = st.sidebar.text_input("Parquet file path", DATA_PATH_DEFAULT)

    if not data_path:
        st.warning("Please provide a parquet file path.")
        st.stop()

    try:
        years = get_available_years(data_path)
    except Exception as e:
        st.error(f"Error reading years from parquet: {e}")
        st.stop()

    active_years = st.sidebar.multiselect("Active years", years, default=years)
    if not active_years:
        active_years = years

    st.sidebar.markdown("### Focus period (for top-K reports)")
    use_focus = st.sidebar.checkbox("Use focus period", True)
    if use_focus:
        start_year = st.sidebar.selectbox("Focus start year", active_years, index=0)
        start_month = st.sidebar.slider("Focus start month", 1, 12, 1)
        end_year = st.sidebar.selectbox("Focus end year", active_years, index=len(active_years) - 1)
        end_month = st.sidebar.slider("Focus end month", 1, 12, 12)
        start_ym = f"{start_year}-{start_month:02d}"
        end_ym = f"{end_year}-{end_month:02d}"
    else:
        start_ym = f"{min(active_years)}-01"
        end_ym = f"{max(active_years)}-12"

    export_figs = st.sidebar.checkbox("Export all charts as JPG", False)

    st.sidebar.markdown("### Top-K settings")
    top_k_global = st.sidebar.number_input("Top K (global, ignored now, using 3)", 1, 20, 3)
    top_k_partners = st.sidebar.number_input("Top K partners in bar chart", 1, 50, 10)
    top_k_partner_categories = st.sidebar.number_input("Top K categories per partner", 1, 20, 3)
    top_k_merchants_in_cat = st.sidebar.number_input("Top K merchants per category", 1, 20, 5)

    st.sidebar.markdown("### Reports to show")
    run_basic_sample = st.sidebar.checkbox("Basic sample", True)
    run_unique_counts = st.sidebar.checkbox("Unique counts by year", True)
    run_global_tops = st.sidebar.checkbox("Top 3 entities in focus", True)
    run_members_per_year = st.sidebar.checkbox("Members vs anonymous per year", True)
    run_category_pies = st.sidebar.checkbox("Category pies & focus comparison", True)
    run_top_partners = st.sidebar.checkbox("Top partners by members", True)
    run_paid_level = st.sidebar.checkbox("Views by paid level", True)
    run_top3_merchants_per_category_flag = st.sidebar.checkbox("Top 3 merchants per category (focus)", True)
    run_time_trends_flag = st.sidebar.checkbox("Time based trends", True)
    run_session_report = st.sidebar.checkbox("Session reports", False)

    st.sidebar.markdown("### Drilldown")
    partner_id_input = st.sidebar.text_input("Partner ID for drilldown", "")
    category_id_input = st.sidebar.text_input("Category ID for drilldown", "")

    st.write(f"**Active years:** {active_years}")
    if use_focus:
        st.write(f"**Focus period:** {start_ym} → {end_ym}")
    else:
        st.write("**Focus period disabled**, using full active years for top-K reports.")

    anon_col = "member_id_str" if has_column(data_path, "member_id_str") else "ip"

    if run_basic_sample:
        show_basic_sample(data_path)

    if run_unique_counts:
        unique_counts_by_year(data_path, active_years, anon_col)

    if run_global_tops:
        show_top3_focus_bars(data_path, active_years, start_ym, end_ym, export_figs)

    if run_members_per_year:
        members_per_year(data_path, active_years, anon_col, export_figs)

    if run_category_pies:
        category_pies_and_focus(data_path, active_years, start_ym, end_ym, export_figs)

    if run_top_partners:
        top_partners_focus(data_path, active_years, start_ym, end_ym, top_k_partners, export_figs)

    if run_paid_level:
        paid_level_analysis(data_path, active_years, export_figs)

    if run_top3_merchants_per_category_flag:
        top3_merchants_per_category(data_path, active_years, start_ym, end_ym)

    if run_time_trends_flag:
        time_trends(data_path, active_years, export_figs)

    if run_session_report:
        session_reports(data_path, active_years, export_figs)

    st.header("Drilldown reports")

    col_p, col_c = st.columns(2)

    with col_p:
        st.subheader("Partner drilldown")
        if partner_id_input.strip():
            try:
                pid = int(float(partner_id_input))
                partner_drilldown(data_path, pid, top_k_partner_categories)
            except ValueError:
                st.warning("Partner ID must be a number.")

    with col_c:
        st.subheader("Category drilldown")
        if category_id_input.strip():
            try:
                cid = int(float(category_id_input))
                category_drilldown(data_path, cid, top_k_merchants_in_cat, active_years)
            except ValueError:
                st.warning("Category ID must be numeric for this version.")


if __name__ == "__main__":
    main()

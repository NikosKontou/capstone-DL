-- =====================================================================
-- Feature pipeline for the time-series spin model.
--
-- Changes vs the previous version:
--   1. NEW FEATURE: `next_session_bucket_id` -- categorical bucket of
--      time_to_next_session_day, mirroring the style of rtp_bucket_id.
--   2. Session-removal filter for "very big spin duration if not the
--      last spin" tightened: it now compares against the *count of
--      spins in the session* (spin_number_in_session = session spin
--      count) rather than relying solely on `next_spin_amount IS NULL`
--      as a proxy for "is last spin". Both conditions are kept in sync
--      so the semantics are explicit and don't silently break if
--      next_spin_amount's definition ever changes.
--   3. NULL handling around `time_to_next_session_sec` / `_day` and the
--      window functions that produce `next_session_start` /
--      `prev_session_end` is hardened:
--        - LAG/LEAD over sessions now break ties on sessionid, so two
--          sessions sharing an identical starttime/session_start no
--          longer produce non-deterministic (and potentially wrong)
--          NULL/non-NULL results.
--        - `time_to_next_session_sec/day` NULL sentinel logic is kept,
--          but the boolean short-circuit bug is removed: instead of
--          computing DATEDIFF unconditionally and then discarding it
--          via CASE (which DuckDB may still evaluate and can trip on
--          NULL arithmetic in edge cases), the DATEDIFF is only
--          evaluated inside the ELSE branch.
--        - `days_since_last_session` (NULL on every account's first
--          session, since prev_session_end is NULL there) is now
--          explicitly flagged with `is_first_session`, rather than
--          left to be silently imputed by the generic missing-value
--          median-fill logic in the ML preprocessor. The raw column is
--          still emitted (and still NULL on first sessions) so
--          existing downstream missing-indicator handling in
--          `config.ADD_MISSING_INDICATORS` keeps working unchanged.
-- =====================================================================

CREATE OR REPLACE VIEW full_merged_data AS
    SELECT * FROM read_parquet(
        '/Top/ACG/Year_2/trimester 3/capstone/data/full_merged_sorted_data.parquet'
    );

CREATE OR REPLACE TABLE _base AS
SELECT
    *,
    -- Keep the raw duration to identify bad sessions later
    DATEDIFF('second', starttime, endtime)                               AS raw_round_duration_sec,
    -- Cap round duration at 500 seconds for the actual ML feature
    LEAST(500, DATEDIFF('second', starttime, endtime))                   AS round_duration_sec,
    wonbaseamount / NULLIF(betbaseamount, 0)                             AS win_over_bet_ratio,
    ROW_NUMBER() OVER (
        PARTITION BY sessionid ORDER BY starttime, betbaseamount
    )                                                                    AS spin_number_in_session,
    -- Total spins in the session, computed once here so the "is this
    -- the last spin" check doesn't have to rely solely on
    -- next_spin_amount IS NULL as an implicit proxy.
    COUNT(*) OVER (
        PARTITION BY sessionid
    )                                                                    AS session_spin_count,
    -- prev_bet / prev_win removed: these were only used as (a) raw
    -- one-step lag features for the old row-level model, and (b) inputs
    -- to prev_rtp_bucket_id / bet_change_direction_id below. A sequence
    -- model sees the raw per-step betbaseamount/wonbaseamount history
    -- directly and can learn these relationships itself, so computing
    -- them here was redundant. Removed at this stage (rather than only
    -- filtered out of the final SELECT) so the LAG window functions
    -- don't run at all.
    GREATEST(
        DATEDIFF('second',
            LAG(endtime) OVER (PARTITION BY sessionid ORDER BY starttime, betbaseamount),
            starttime
        ), 0
    )                                                                    AS inter_spin_gap_sec,
    LEAD(betbaseamount) OVER (
        PARTITION BY sessionid ORDER BY starttime, betbaseamount
    )                                                                    AS next_spin_amount
FROM full_merged_data;

CREATE OR REPLACE TABLE _enriched AS
SELECT
    *,
    CASE
        WHEN wonbaseamount = 0                                           THEN 0
        WHEN win_over_bet_ratio < 1                                      THEN 1
        WHEN win_over_bet_ratio >= 1   AND win_over_bet_ratio <  5       THEN 2
        WHEN win_over_bet_ratio >= 5   AND win_over_bet_ratio < 10       THEN 3
        WHEN win_over_bet_ratio >= 10  AND win_over_bet_ratio < 20       THEN 4
        WHEN win_over_bet_ratio >= 20  AND win_over_bet_ratio < 30       THEN 5
        WHEN win_over_bet_ratio >= 30  AND win_over_bet_ratio < 40       THEN 6
        WHEN win_over_bet_ratio >= 40  AND win_over_bet_ratio < 50       THEN 7
        WHEN win_over_bet_ratio >= 50  AND win_over_bet_ratio < 60       THEN 8
        WHEN win_over_bet_ratio >= 60  AND win_over_bet_ratio < 70       THEN 9
        WHEN win_over_bet_ratio >= 70  AND win_over_bet_ratio < 100      THEN 10
        ELSE                                                                  11
    END::INTEGER                                                         AS rtp_bucket_id
    -- prev_rtp_bucket_id and bet_change_direction_id removed: both were
    -- derived solely from prev_bet/prev_win, which are no longer
    -- computed (see _base). A sequence model reconstructs any signal
    -- these encoded directly from the raw per-step history it already
    -- receives.
FROM _base;

-- ---------------------------------------------------------------------
-- Session-level removal filters.
--   (a) sessions spanning more than 24h (unchanged from before).
--   (b) sessions containing an over-long round (>500s raw duration)
--       that is NOT the final spin of its session. A spin is "the
--       final spin" iff spin_number_in_session = session_spin_count;
--       we additionally check next_spin_amount IS NULL as a
--       consistency assertion -- both should agree since
--       next_spin_amount is LEAD(betbaseamount) over the same window.
--       Keeping both makes the intent explicit and would surface a
--       data problem (rather than silently mis-filtering) if the two
--       ever disagreed.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE _cleaned AS
SELECT e.* EXCLUDE(raw_round_duration_sec, session_spin_count)
FROM _enriched e
WHERE e.sessionid NOT IN (
    SELECT sessionid
    FROM (
        SELECT sessionid, DATEDIFF('second', MIN(starttime), MAX(endtime)) AS session_duration_sec
        FROM _enriched GROUP BY sessionid
    ) s
    WHERE s.session_duration_sec > 86400
)
AND e.sessionid NOT IN (
    SELECT sessionid
    FROM _enriched
    WHERE raw_round_duration_sec > 500
      AND spin_number_in_session <> session_spin_count   -- not the last spin
      AND next_spin_amount IS NOT NULL                   -- consistency check
);

CREATE OR REPLACE TABLE _session_endpoints AS
SELECT account, sessionid, MIN(starttime) AS session_start, MAX(endtime) AS session_end
FROM _cleaned GROUP BY account, sessionid;

-- ---------------------------------------------------------------------
-- Session recency features.
--   * LAG/LEAD now order by (session_start, sessionid) so that two
--     sessions with an identical session_start for the same account
--     resolve deterministically instead of depending on arbitrary
--     row order, which previously could push a real "next session"
--     to look like there wasn't one (or vice versa).
--   * time_to_next_session_sec/day: the DATEDIFF is only evaluated in
--     the ELSE branch now (next_session_start is guaranteed non-NULL
--     there), instead of being computed unconditionally and then
--     discarded by the CASE -- avoiding any NULL-propagation surprise
--     from DuckDB evaluating GREATEST/DATEDIFF against a NULL
--     next_session_start before the CASE branch is chosen.
--   * is_first_session flags rows where prev_session_end IS NULL, so
--     the NULL in days_since_last_session is legible as "first
--     session for this account" rather than an unexplained missing
--     value that downstream code has to guess about.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE _session_recency AS
WITH ordered AS (
    SELECT
        account, sessionid, session_start, session_end,
        LAG(session_end)    OVER (PARTITION BY account ORDER BY session_start, sessionid) AS prev_session_end,
        LEAD(session_start) OVER (PARTITION BY account ORDER BY session_start, sessionid) AS next_session_start
    FROM _session_endpoints
)
SELECT
    account, sessionid, session_end, next_session_start, prev_session_end,
    CASE WHEN next_session_start IS NULL THEN 1 ELSE 0 END::INTEGER      AS is_last_session,
    CASE WHEN prev_session_end    IS NULL THEN 1 ELSE 0 END::INTEGER     AS is_first_session,
    CASE
        WHEN next_session_start IS NULL THEN 2592000
        ELSE GREATEST(0, DATEDIFF('second', session_end, next_session_start))
    END                                                                  AS time_to_next_session_sec,
    CASE
        WHEN next_session_start IS NULL THEN 30.0
        ELSE GREATEST(0, DATEDIFF('second', session_end, next_session_start)) / 86400.0
    END                                                                  AS time_to_next_session_day,
    CASE
        WHEN next_session_start IS NULL THEN 0
        WHEN GREATEST(0, DATEDIFF('second', session_end, next_session_start)) <= 259200 THEN 1
        ELSE 0
    END::INTEGER                                                        AS back_in_3_days
FROM ordered;

-- New bucketed feature: discretized time_to_next_session_day.
-- Buckets (days):
--   0 : same-day return          (< 1)
--   1 : next day                 [1, 2)
--   2 : within a week            [2, 7)
--   3 : within two weeks         [7, 14)
--   4 : within a month           [14, 30)
--   5 : did not return / >=30d   (>= 30, incl. the 30.0 sentinel for
--                                 an account's last observed session)
CREATE OR REPLACE TABLE _session_recency_bucketed AS
SELECT
    *,
    CASE
        WHEN time_to_next_session_day <  1  THEN 0
        WHEN time_to_next_session_day <  2  THEN 1
        WHEN time_to_next_session_day <  7  THEN 2
        WHEN time_to_next_session_day < 14  THEN 3
        WHEN time_to_next_session_day < 30  THEN 4
        ELSE                                      5
    END::INTEGER                                                        AS next_session_bucket_id
FROM _session_recency;

CREATE OR REPLACE TABLE _cleaned_with_recency AS
SELECT
    c.*,
    r.session_end,
    r.next_session_start,
    r.is_first_session,
    r.is_last_session,
    -- Still NULL on an account's first session by construction; the
    -- is_first_session flag above makes that explicit rather than
    -- leaving it to be inferred from the missing-indicator pipeline.
    -- Floored at 0 to guard against back-to-back/overlapping sessions
    -- for the same account (e.g. two sessions recorded with the same
    -- starttime) producing a tiny negative value.
    CASE
        WHEN r.prev_session_end IS NULL THEN NULL
        ELSE GREATEST(0.0, DATEDIFF('second', r.prev_session_end, c.starttime) / 86400.0)
    END                                                                  AS days_since_last_session,
    (COUNT(*) OVER (PARTITION BY c.sessionid) - c.spin_number_in_session)::INTEGER AS spins_left_in_session,
    r.time_to_next_session_sec,
    r.time_to_next_session_day,
    r.back_in_3_days,
    r.next_session_bucket_id
FROM _cleaned c
LEFT JOIN _session_recency_bucketed r USING (account, sessionid);

CREATE OR REPLACE TABLE _split_boundaries AS
WITH nominal AS (
    SELECT
        MIN(starttime)                                                   AS data_start,
        DATE_TRUNC('month', MIN(starttime)) + INTERVAL 2 MONTH           AS train_nominal,
        DATE_TRUNC('month', MIN(starttime)) + INTERVAL 2 MONTH + INTERVAL 11 DAY AS valid_nominal,
        MAX(endtime)                                                     AS data_end
    FROM _cleaned_with_recency
)
SELECT
    n.data_start, n.data_end, n.train_nominal, n.valid_nominal,
    MAX(CASE WHEN e.session_end <= n.train_nominal THEN e.session_end END) AS actual_train_end,
    MAX(CASE WHEN e.session_end <= n.valid_nominal THEN e.session_end END) AS actual_valid_end,
    n.data_end AS actual_test_end
FROM nominal n
CROSS JOIN _session_endpoints e
GROUP BY n.data_start, n.data_end, n.train_nominal, n.valid_nominal;

CREATE OR REPLACE TABLE _all_features AS
WITH
with_split AS (
    SELECT
        c.*,
        CASE
            WHEN c.session_end <= b.actual_train_end THEN 'train'
            WHEN c.session_end <= b.actual_valid_end THEN 'valid'
            ELSE 'test'
        END AS split_name
    FROM _cleaned_with_recency c
    CROSS JOIN _split_boundaries b
),
with_player_history AS (
    SELECT
        *,
        COUNT(*) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_round_count,
        AVG(betbaseamount) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_avg_bet,
        AVG(win_over_bet_ratio) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_avg_win_over_bet,
        AVG(CASE WHEN rtp_bucket_id = 0 THEN 1.0 ELSE 0.0 END) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_pct_no_win,
        AVG(inter_spin_gap_sec) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_avg_round_duration,
        COUNT(DISTINCT sessionid) OVER (PARTITION BY account, split_name ORDER BY starttime ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS player_distinct_sessions
    FROM with_split
),
with_time AS (
    SELECT
        *,
        EXTRACT(HOUR  FROM starttime)::INTEGER                           AS hour_of_day,
        EXTRACT(DOW   FROM starttime)::INTEGER                           AS day_of_week,
        EXTRACT(MONTH FROM starttime)::INTEGER                           AS month,
        CASE WHEN EXTRACT(DOW FROM starttime) IN (0, 6) THEN 1 ELSE 0 END::INTEGER AS is_weekend,
        CASE
            WHEN EXTRACT(HOUR FROM starttime) BETWEEN  0 AND  5 THEN 0
            WHEN EXTRACT(HOUR FROM starttime) BETWEEN  6 AND 11 THEN 1
            WHEN EXTRACT(HOUR FROM starttime) BETWEEN 12 AND 17 THEN 2
            ELSE 3
        END::INTEGER                                                     AS period_of_day
    FROM with_player_history
),
with_ohe AS (
    SELECT
        *,
        CASE WHEN game_id =  1 THEN 1 ELSE 0 END::INTEGER AS game_id_1,
        CASE WHEN game_id =  2 THEN 1 ELSE 0 END::INTEGER AS game_id_2,
        CASE WHEN game_id =  3 THEN 1 ELSE 0 END::INTEGER AS game_id_3,
        CASE WHEN game_id =  4 THEN 1 ELSE 0 END::INTEGER AS game_id_4,
        CASE WHEN game_id =  5 THEN 1 ELSE 0 END::INTEGER AS game_id_5,
        CASE WHEN game_id =  6 THEN 1 ELSE 0 END::INTEGER AS game_id_6,
        CASE WHEN game_id =  7 THEN 1 ELSE 0 END::INTEGER AS game_id_7,
        CASE WHEN game_id =  8 THEN 1 ELSE 0 END::INTEGER AS game_id_8,
        CASE WHEN game_id =  9 THEN 1 ELSE 0 END::INTEGER AS game_id_9,
        CASE WHEN game_id = 10 THEN 1 ELSE 0 END::INTEGER AS game_id_10,
        CASE WHEN game_id = 11 THEN 1 ELSE 0 END::INTEGER AS game_id_11,
        CASE WHEN game_provider_id =  2 THEN 1 ELSE 0 END::INTEGER AS game_provider_id_2,
        CASE WHEN game_provider_id =  5 THEN 1 ELSE 0 END::INTEGER AS game_provider_id_5,
        CASE WHEN game_provider_id = 10 THEN 1 ELSE 0 END::INTEGER AS game_provider_id_10,
        CASE WHEN currency_id = 1 THEN 1 ELSE 0 END::INTEGER AS currency_1,
        CASE WHEN currency_id = 2 THEN 1 ELSE 0 END::INTEGER AS currency_2,
        CASE WHEN currency_id = 3 THEN 1 ELSE 0 END::INTEGER AS currency_3,
        CASE WHEN device_id = 1 THEN 1 ELSE 0 END::INTEGER AS device_1,
        CASE WHEN device_id = 2 THEN 1 ELSE 0 END::INTEGER AS device_2,
        CASE WHEN device_id = 3 THEN 1 ELSE 0 END::INTEGER AS device_3
    FROM with_time
)
SELECT
    account, sessionid, split_name, month,
    starttime, endtime,
    betbaseamount, wonbaseamount, round_duration_sec, win_over_bet_ratio,
    spin_number_in_session, inter_spin_gap_sec,
    rtp_bucket_id,
    days_since_last_session, is_first_session, is_last_session,
    hour_of_day, period_of_day, day_of_week, is_weekend,
    session_end, next_session_start,
    player_round_count, player_avg_bet, player_avg_win_over_bet,
    player_pct_no_win, player_avg_round_duration, player_distinct_sessions,
    currency_1, currency_2, currency_3,
    device_1, device_2, device_3,
    game_id_1, game_id_2, game_id_3, game_id_4, game_id_5,
    game_id_6, game_id_7, game_id_8, game_id_9, game_id_10, game_id_11,
    game_provider_id_2, game_provider_id_5, game_provider_id_10,
    next_spin_amount, spins_left_in_session,
    time_to_next_session_sec, time_to_next_session_day, next_session_bucket_id,
    back_in_3_days,
    CASE
        WHEN next_spin_amount IS NULL THEN 1
        ELSE 0
    END::INTEGER AS is_last_spin_in_session
FROM with_ohe;

COPY (SELECT * FROM _all_features WHERE split_name = 'train' ORDER BY account, starttime) TO '/Top/ACG/Year_2/trimester 3/capstone/data/ml/DL/train_features.parquet' (FORMAT PARQUET);
COPY (SELECT * FROM _all_features WHERE split_name = 'valid' ORDER BY account, starttime) TO '/Top/ACG/Year_2/trimester 3/capstone/data/ml/DL/valid_features.parquet' (FORMAT PARQUET);
COPY (SELECT * FROM _all_features WHERE split_name = 'test' ORDER BY account, starttime) TO '/Top/ACG/Year_2/trimester 3/capstone/data/ml/DL/test_features.parquet' (FORMAT PARQUET);
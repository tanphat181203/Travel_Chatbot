import os
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_ENDPOINT_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conn_pool = None
try:
    if DB_ENDPOINT_ID:
        DATABASE_URL = (
            f"postgresql://{DB_USER}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            f"?sslmode=require"
        )
        logger.info(f"Connecting to database using hostname with embedded project info")
    else:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        logger.info("Connecting to database using standard host/port.")


    logger.info("Initializing database connection pool...")
    conn_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL
    )


    logger.info("Database connection pool initialized successfully.")

except (psycopg2.OperationalError, Exception) as e:
    logger.error(f"FATAL: Failed to initialize database connection pool: {e}", exc_info=True)
    conn_pool = None

@contextmanager
def get_pooled_connection():
    if conn_pool is None:
        logger.error("Connection pool is not available.")
        raise ConnectionError("Database connection pool is not initialized.")

    conn = None
    try:
        conn = conn_pool.getconn()
        logger.debug(f"Acquired connection {id(conn)} from pool.")

        yield conn
        conn.commit()
        logger.debug(f"Committed transaction on connection {id(conn)}.")
    except (Exception, psycopg2.Error) as e:
        logger.error(f"Error during database operation with connection {id(conn)}: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                logger.debug(f"Rolled back transaction on connection {id(conn)}.")
            except psycopg2.Error as rb_err:
                logger.error(f"Error during rollback on connection {id(conn)}: {rb_err}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                conn_pool.putconn(conn)
                logger.debug(f"Returned connection {id(conn)} to pool.")
            except Exception as pc_err:
                 logger.error(f"Error returning connection {id(conn)} to pool: {pc_err}", exc_info=True)


def execute_query(query: str, params: tuple = None, fetch_one: bool = False):
    logger.debug(f"Executing query: {query[:150]}... with params: {params}")
    if conn_pool is None:
         logger.error("Cannot execute query: Connection pool not available.")
         return None

    results = None
    try:
        with get_pooled_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)

                if fetch_one:
                    result_dict = cur.fetchone()
                    logger.debug(f"Fetch one result: {result_dict}")
                    results = dict(result_dict) if result_dict else None
                else:
                    results_list = cur.fetchall()
                    results = [dict(row) for row in results_list]
                    logger.debug(f"Fetch all results (count: {len(results)}): {results[:3]}")

        return results

    except ConnectionError as e:
        logger.error(f"Failed to get connection from pool: {e}", exc_info=True)
        return None
    except psycopg2.Error as e:
        logger.error(f"Database query error: {e} (SQLSTATE: {e.pgcode})", exc_info=True)
        logger.error(f"Failed Query (first 150 chars): {query[:150]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during query execution: {e}", exc_info=True)
        logger.error(f"Failed Query (first 150 chars): {query[:150]}...")
        return None


def get_available_locations():
    query = """
        SELECT DISTINCT unnest(destination) AS destination
        FROM Tour
        WHERE availability = true
        ORDER BY destination;
    """
    results = execute_query(query)
    if results:
        return [row['destination'] for row in results]
    elif results == []:
        return []
    else:
        logger.error("Failed to fetch available locations due to database error.")
        return None

def get_tour_by_id(tour_id):
    query = """
    SELECT
        t.tour_id,
        t.title,
        t.duration,
        t.departure_location,
        t.destination,
        t.region,
        t.itinerary,
        t.max_participants,
        d.departure_id,
        d.start_date,
        d.price_adult,
        d.price_child_120_140,
        d.price_child_100_120,
        p.promotion_id,
        p.name AS promotion_name,
        p.type AS promotion_type,
        p.discount AS promotion_discount,
        p.start_date AS promotion_start_date,
        p.end_date AS promotion_end_date
    FROM Tour t
    LEFT JOIN Departure d ON t.tour_id = d.tour_id AND d.availability = true
    LEFT JOIN Tour_Promotion tp ON t.tour_id = tp.tour_id
    LEFT JOIN Promotion p ON tp.promotion_id = p.promotion_id
        AND CURRENT_DATE BETWEEN p.start_date AND p.end_date
        AND p.status = 'active'
    WHERE t.tour_id = %s AND t.availability = true
    ORDER BY d.start_date
    LIMIT 1;
    """

    result = execute_query(query, (tour_id,), fetch_one=True)
    if result is None:
        logger.warning(f"Tour with ID {tour_id} not found or database error occurred.")
    else:
        logger.info(f"Successfully retrieved tour with ID {tour_id}")

    return result

def search_tours_db(entities: dict):
    base_query = """
    SELECT
        t.tour_id,
        t.title,
        t.duration,
        t.departure_location,
        t.destination,
        t.region,
        t.itinerary,
        t.max_participants,
        d.departure_id,
        d.start_date,
        d.price_adult,
        d.price_child_120_140,
        d.price_child_100_120,
        p.promotion_id,
        p.name AS promotion_name,
        p.type AS promotion_type,
        p.discount AS promotion_discount,
        p.start_date AS promotion_start_date,
        p.end_date AS promotion_end_date
    FROM Departure d
    JOIN Tour t ON d.tour_id = t.tour_id
    LEFT JOIN Tour_Promotion tp ON t.tour_id = tp.tour_id
    LEFT JOIN Promotion p ON tp.promotion_id = p.promotion_id
        AND d.start_date BETWEEN p.start_date AND p.end_date
        AND p.status = 'active'
    WHERE t.availability = true AND d.availability = true
    """
    filters = []
    params = []


    if entities.get('region'):
        filters.append("t.region = %s")
        params.append(entities['region'])


    if entities.get('destination'):
        dest_list = entities['destination'] if isinstance(entities['destination'], list) else [entities['destination']]
        filters.append("t.destination && %s::text[]")
        params.append(dest_list)


    if entities.get('duration'):
        filters.append("t.duration ILIKE %s")
        params.append(f"%{entities['duration']}%")


    if entities.get('time'):
        time_filter_parts = []
        time_info = entities['time']
        if not isinstance(time_info, list): time_info = [time_info]
        for time_obj in time_info:
            if 'departure_date' in time_obj:
                time_filter_parts.append("d.start_date = %s")
                params.append(time_obj['departure_date'])
            elif 'start_date' in time_obj and 'end_date' in time_obj:
                time_filter_parts.append("d.start_date BETWEEN %s AND %s")
                params.extend([time_obj['start_date'], time_obj['end_date']])
        if time_filter_parts: filters.append(f"({' OR '.join(time_filter_parts)})")


    if entities.get('budget'):
        budget = str(entities['budget'])
        try:
            if '-' in budget:
                min_price, max_price = map(float, budget.split('-'))
                filters.append("d.price_adult BETWEEN %s AND %s")
                params.extend([min_price, max_price])
            else:
                max_price = float(budget)
                filters.append("d.price_adult <= %s")
                params.append(max_price)
        except ValueError: logger.warning(f"Could not parse budget: {budget}")


    if entities.get('number_of_people'):
        num_people = str(entities['number_of_people'])
        min_required = 1
        try:
            if num_people.startswith('>'): min_required = int(num_people[1:]) + 1
            elif '-' in num_people: min_req, _ = map(int, num_people.split('-')); min_required = max(min_required, min_req)
            else: min_required = max(min_required, int(num_people))
        except ValueError: pass
        if min_required > 1:
            filters.append("t.max_participants >= %s")
            params.append(min_required)


    if filters:
        base_query += " AND " + " AND ".join(filters)

    base_query += " ORDER BY d.start_date, t.title;"


    results = execute_query(base_query, tuple(params))

    if results is None:
        logger.error("Failed to search tours due to database error.")
        return []
    elif not results:
         logger.info("No tours found matching the criteria.")
         return []
    else:
        logger.info(f"Found {len(results)} results from database.")
        return results
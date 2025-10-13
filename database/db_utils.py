import psycopg2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def connect_to_database():
    """Connect to the database"""
    return psycopg2.connect(
        database='gamuda_weld_db',
        user='jeremyong',
        host='localhost',
        port='5432'
    )

def get_projects_from_database():
    """Fetch project locations from PostgreSQL database"""
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute("SELECT id, location FROM project")
        rows = cursor.fetchall()
        
        # Format as list of strings for dropdown
        project_choices = [f"{row[0]} - {row[1]}" for row in rows]
        
        cursor.close()
        conn.close()
        
        logger.info(f"Loaded {len(project_choices)} projects from database")
        return project_choices
        
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return ["Database connection failed"]

def get_project_details(selected_project):
    """Get detailed project information from database based on selected project"""
    if not selected_project or selected_project == "Database connection failed":
        return "", "", "", "", "", ""
    
    try:
        # Extract project ID from the dropdown selection (format: "ID - Location")
        project_id = selected_project.split(" - ")[0]
        logger.info(f"Auto-filling project details for ID: {project_id}")
        
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT location, start_date, end_date, description, created_by 
            FROM project 
            WHERE id = %s
        """, (project_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            location, start_date, end_date, description, created_by = row
            
            # Convert dates to full datetime string format for Gradio DateTime
            start_date_str = start_date.strftime("%Y-%m-%d 00:00:00") if start_date else ""
            end_date_str = end_date.strftime("%Y-%m-%d 00:00:00") if end_date else ""
            
            return (location, start_date_str, end_date_str, description, created_by, "In Progress")
            
        else:
            return "", "", "", "", "", ""
            
    except Exception as e:
        logger.error(f"Error fetching project details: {e}")
        return "", "", "", "", "", ""

def push_results_to_database(project_id, created_by, description, good_weld, bad_weld, uncertain):
    """Push the weld quality results to the database
    
    Args:
        project_id (int): ID of the project
        created_by (str): Name of person who created the report
        description (str): Description of the inspection
        good_weld (int): Number of good welds
        bad_weld (int): Number of bad welds
        uncertain (int): Number of uncertain/low confidence welds
    """
    try:
        # Validate required fields
        if not project_id or not created_by:
            logger.error("project_id and created_by are required")
            return False
            
        # Ensure numeric fields are integers
        try:
            project_id = int(project_id)
            good_weld = int(good_weld)
            bad_weld = int(bad_weld)
            uncertain = int(uncertain)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid numeric value: {e}")
            return False
            
        # Truncate strings to match database limits
        created_by = str(created_by)[:255]
        description = str(description)[:255] if description else None
        
        conn = connect_to_database()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO weld_quality 
                (project_id, created_by, description, good_weld, bad_weld, uncertain)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (project_id, created_by, description, good_weld, bad_weld, uncertain))
        
        conn.commit()
        logger.info(f"Successfully pushed results to database for project {project_id}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error pushing results to database: {e}")
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass
        return False

def get_specific_results(project_id, created_by):
    """
    Returns the specific results for a given project_id and created_by
    
    Returns:
        List of dictionaries containing:
            - id: The record ID
            - project_id: The project ID
            - created_by: Who created the record
            - description: The QA description
            - good_weld: Number of good welds
            - bad_weld: Number of bad welds
            - uncertain: Number of uncertain welds
    """
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, project_id, created_by, description, good_weld, bad_weld, uncertain,
                   to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at
            FROM weld_quality 
            WHERE project_id = %s AND created_by = %s 
            ORDER BY created_at DESC, id DESC
        """, (project_id, created_by))
        
        # Convert rows to list of dictionaries
        results = []
        for row in cursor.fetchall():
            result = {
                'id': row[0],
                'project_id': row[1],
                'created_by': row[2],
                'description': row[3] or 'Not specified',
                'good_weld': row[4],
                'bad_weld': row[5],
                'uncertain': row[6],
                'created_at': row[7],
                'total_welds': row[4] + row[5] + row[6]  # Sum of all welds
            }
            results.append(result)
            
        cursor.close()
        conn.close()
        return results
    
    except Exception as e:
        logger.error(f"Error getting specific results: {e}")


def get_users_from_database():
    """Get list of users from the users table"""
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users ORDER BY name")  # Order by ID to maintain consistent order
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Return list of user names
        users = [row[0] for row in rows]  # Each row[0] is the name column
        logger.info(f"Found users in database: {users}")
        return users
        
    except Exception as e:
        logger.error(f"Error getting users from database: {e}")

def get_reports_from_database():
    """Get list of reports from the database"""
    
    conn = connect_to_database()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, project_id, created_by, description, good_weld, bad_weld, uncertain,
            to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at
    FROM weld_quality 
    """)


    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} reports")
    
    cursor.close()
    conn.close()
    return rows

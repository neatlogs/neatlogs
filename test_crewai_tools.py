"""
Test CrewAI with various tool types to see what OpenInference captures.

This file tests:
1. HTTP API calls (via requests)
2. Subprocess calls
3. Custom Python functions
4. File I/O operations
5. Database calls (local Postgres/ClickHouse)
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from typing import Type

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    tags=["v4", "crewai", "tool-testing"],
    workflow_name="crewai-tool-test"
)

# ============================================================================
# TOOL 1: HTTP API Call (Should be auto-captured by requests instrumentor)
# ============================================================================
class WeatherInput(BaseModel):
    """Input for weather tool."""
    city: str = Field(..., description="City name to fetch weather for")

class FetchWeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = "Fetch weather data from an external API for a given city"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        """Fetch weather data from external API."""
        print(f"[Tool] Fetching weather for {city}...")
        try:
            # This HTTP call should be auto-captured if requests is instrumented
            response = requests.get(
                f"https://wttr.in/{city}?format=j1",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                temp = data['current_condition'][0]['temp_C']
                weather = data['current_condition'][0]['weatherDesc'][0]['value']
                return f"Weather in {city}: {temp}°C, {weather}"
            else:
                return f"Failed to fetch weather (status: {response.status_code})"
        except Exception as e:
            return f"Error fetching weather: {str(e)}"


# ============================================================================
# TOOL 2: Subprocess Call (NOT auto-captured)
# ============================================================================
class CommandInput(BaseModel):
    """Input for shell command tool."""
    command: str = Field(..., description="Shell command to execute")

class RunShellCommandTool(BaseTool):
    name: str = "run_shell_command"
    description: str = "Run a shell command and return output"
    args_schema: Type[BaseModel] = CommandInput
    
    def _run(self, command: str) -> str:
        """Run shell command."""
        print(f"[Tool] Running shell command: {command}")
        try:
            # Subprocess calls are NOT auto-captured by OpenTelemetry
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=5
            )
            return f"Exit code: {result.returncode}\nOutput: {result.stdout}"
        except Exception as e:
            return f"Error running command: {str(e)}"


# ============================================================================
# TOOL 3: Custom Python Function (NOT auto-captured)
# ============================================================================
class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    base_price: float = Field(..., description="Base price amount")
    tax_rate: float = Field(..., description="Tax rate as decimal (e.g., 0.08 for 8%)")

class CalculatePriceTool(BaseTool):
    name: str = "calculate_price"
    description: str = "Calculate final price with tax"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, base_price: float, tax_rate: float) -> str:
        """Calculate price with tax."""
        print(f"[Tool] Calculating price: base=${base_price}, tax={tax_rate}")
        # Pure Python logic - NOT captured unless we manually add spans
        final_price = base_price * (1 + tax_rate)
        return f"Final price: ${final_price:.2f}"


# ============================================================================
# TOOL 4: File I/O (NOT auto-captured)
# ============================================================================
class FileInput(BaseModel):
    """Input for file read tool."""
    filename: str = Field(..., description="Path to file to read")

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read contents of a file"
    args_schema: Type[BaseModel] = FileInput
    
    def _run(self, filename: str) -> str:
        """Read file contents."""
        print(f"[Tool] Reading file: {filename}")
        try:
            # File I/O is NOT auto-captured
            path = Path(filename)
            if path.exists():
                content = path.read_text()
                return f"File contents ({len(content)} chars): {content[:100]}..."
            else:
                return f"File not found: {filename}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


# ============================================================================
# TOOL 5: Local Database Query (Postgres - MAY be auto-captured with instrumentor)
# ============================================================================
class DatabaseQueryInput(BaseModel):
    """Input for database query tool."""
    query: str = Field(..., description="SQL query to execute")

class QueryDatabaseTool(BaseTool):
    name: str = "query_database"
    description: str = "Execute a SQL query on local Postgres database"
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    def _run(self, query: str) -> str:
        """Execute database query."""
        print(f"[Tool] Querying Postgres database: {query}")
        try:
            import psycopg2
            
            # Connect to local Postgres (NeatLogs database)
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                dbname="neatlogs",
                user="neatlogs",
                password="neatlogs"
            )
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query)
            
            # Fetch results
            if cursor.description:  # SELECT query
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                result = f"Columns: {columns}\nRows ({len(rows)}):\n"
                for row in rows[:5]:  # Limit to first 5 rows
                    result += f"  {row}\n"
                if len(rows) > 5:
                    result += f"  ... and {len(rows) - 5} more rows"
            else:  # INSERT/UPDATE/DELETE
                result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            
            cursor.close()
            conn.close()
            
            return result
        except Exception as e:
            return f"Error querying database: {str(e)}"


# ============================================================================
# TOOL 6: ClickHouse Query (Vector similarity search simulation for RAG)
# ============================================================================
class ClickHouseQueryInput(BaseModel):
    """Input for ClickHouse query tool."""
    search_query: str = Field(..., description="Search query to find similar traces")

class QueryClickHouseTool(BaseTool):
    name: str = "query_clickhouse"
    description: str = "Search ClickHouse for similar traces (RAG-style vector search)"
    args_schema: Type[BaseModel] = ClickHouseQueryInput
    
    def _run(self, search_query: str) -> str:
        """Query ClickHouse for traces."""
        print(f"[Tool] Searching ClickHouse for: {search_query}")
        try:
            # Simple HTTP API call to ClickHouse
            clickhouse_url = "http://localhost:8123"
            
            # Example: Search for traces with specific keywords in span names
            sql_query = f"""
            SELECT 
                trace_id,
                span_name,
                span_type,
                total_tokens,
                duration_ms
            FROM neatlogs.spans
            WHERE span_name ILIKE '%{search_query}%'
            LIMIT 5
            FORMAT JSON
            """
            
            response = requests.post(
                clickhouse_url,
                data=sql_query,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                rows = data.get('data', [])
                if rows:
                    result = f"Found {len(rows)} traces matching '{search_query}':\n"
                    for row in rows:
                        result += f"  - {row['span_name']} ({row['span_type']}) - {row['duration_ms']}ms\n"
                    return result
                else:
                    return f"No traces found matching '{search_query}'"
            else:
                return f"ClickHouse query failed (status: {response.status_code})"
        except Exception as e:
            return f"Error querying ClickHouse: {str(e)}"


# ============================================================================
# CREATE CREWAI AGENTS AND TASKS
# ============================================================================

# Create tool instances
fetch_weather_tool = FetchWeatherTool()
run_shell_tool = RunShellCommandTool()
calculate_price_tool = CalculatePriceTool()
read_file_tool = ReadFileTool()
query_db_tool = QueryDatabaseTool()
query_clickhouse_tool = QueryClickHouseTool()

# Agent 1: Uses HTTP API tool
weather_agent = Agent(
    role="Weather Reporter",
    goal="Fetch and report weather information",
    backstory="You are a weather expert who uses external APIs.",
    tools=[fetch_weather_tool],
    verbose=True
)

# Agent 2: Uses multiple tool types
utility_agent = Agent(
    role="Utility Helper",
    goal="Perform various utility tasks",
    backstory="You can run commands, calculate prices, and read files.",
    tools=[run_shell_tool, calculate_price_tool, read_file_tool],
    verbose=True
)

# Agent 3: Database agent for RAG
database_agent = Agent(
    role="Database Analyst",
    goal="Query databases to retrieve information",
    backstory="You are an expert at querying Postgres and ClickHouse databases.",
    tools=[query_db_tool, query_clickhouse_tool],
    verbose=True
)

# Tasks
task1 = Task(
    description="Get the current weather in San Francisco",
    agent=weather_agent,
    expected_output="Weather report for San Francisco"
)

task2 = Task(
    description="Calculate the final price for an item that costs $100 with 8% tax",
    agent=utility_agent,
    expected_output="Final price with tax"
)

task3 = Task(
    description="List files in the current directory using 'ls -la' command",
    agent=utility_agent,
    expected_output="Directory listing"
)

task4 = Task(
    description="Query the Postgres database to count how many traces exist. Use: SELECT COUNT(*) FROM traces",
    agent=database_agent,
    expected_output="Number of traces in database"
)

task5 = Task(
    description="Search ClickHouse for traces containing 'ChatOpenAI' in their span name",
    agent=database_agent,
    expected_output="List of traces with ChatOpenAI spans"
)

# Create crew
crew = Crew(
    agents=[weather_agent, utility_agent, database_agent],
    tasks=[task1, task2, task3, task4, task5],
    process=Process.sequential,
    verbose=True
)

# ============================================================================
# RUN THE CREW
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting CrewAI Tool Testing...")
    print(f"📍 Backend URL: http://localhost:3000")
    print(f"🔧 Testing 6 tool types:")
    print("   1. HTTP API calls (wttr.in)")
    print("   2. Subprocess (shell commands)")
    print("   3. Custom functions (price calculator)")
    print("   4. File I/O (read file)")
    print("   5. Postgres database queries")
    print("   6. ClickHouse queries (RAG-style)")
    print()
    
    try:
        result = crew.kickoff()
        print("\n✅ Crew execution completed!")
        print(f"\nFinal Result:\n{result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📊 Check your NeatLogs UI to see which tools were captured!")
    print("Expected:")
    print("  ✅ HTTP API calls (fetch_weather) - Captured as HTTP spans")
    print("  ✅ HTTP to ClickHouse (query_clickhouse) - Captured as HTTP spans")
    print("  ⚠️  Postgres queries (query_database) - Captured IF psycopg2 instrumented")
    print("  ❌ Subprocess calls (run_shell_command) - NOT captured")
    print("  ❌ Custom functions (calculate_price) - NOT captured")
    print("  ❌ File I/O (read_file) - NOT captured")

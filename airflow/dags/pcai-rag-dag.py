"""
=============================================================================
SIMPLE RAG TEST PIPELINE FOR KNOWLEDGE BASE
This Airflow DAG runs a mini RAG preprocessing pipeline by checking the input PDF, generating a small set of text chunks, and printing a summary. It demonstrates how tasks pass data through XCom and how intermediate artifacts are saved for reuse. It’s a lightweight end-to-end test of the knowledge-base preparation workflow.
=============================================================================
This DAG has 3 tasks that run in sequence:
1. Check if PDF exists
2. Create 5 sample chunks
3. Print a summary

Tasks share data using XCom (like passing notes)
Tasks save data to files using Pickle (like freezing data)
=============================================================================
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# ============================================================================
# STEP 1: Configure Paths
# ============================================================================
sys.path.append('/Users/nallagat/playground/at-2/pcai-at-2-rag')  

# Import your RAG components
from knowledge_base import KnowledgeBase

# Where is your PDF?
PDF_PATH = '/Users/nallagat/playground/at-2/pcai-at-2-rag/data/hpe-pcai.pdf'  

# Where should output files go?
OUTPUT_DIR = '/Users/nallagat/playground/at-2/pcai-at-2-rag/airflow/test_output'  

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 2: DAG Configuration
# ============================================================================
# Default settings for all tasks
default_args = {
    'owner': 'nallagatla',              # Who owns this DAG
    'retries': 1,               # If task fails, try 1 more time
    'retry_delay': timedelta(minutes=1),  # Wait 1 min before retry
}

# Create the DAG (the pipeline)
dag = DAG(
    dag_id='test_rag_simple',           # Name shown in Airflow UI
    default_args=default_args,          # Use settings above
    description='Simple 3-task test',   # Description
    schedule=None,                       # Manual trigger only (no auto-run)
    start_date=datetime(2024, 1, 1),   # When this DAG became valid
    catchup=False,                       # Don't run for past dates
    tags=['test', 'simple'],            # Tags to organize DAGs
)

# ============================================================================
# TASK 1: CHECK PDF
# ============================================================================

def check_pdf(**context):
    """
    This function checks if the PDF file exists and is readable.
    
    What it does:
    1. Checks if file exists
    2. Gets file size
    3. Saves file size to XCom (so Task 2 can use it)
    
    Args:
        **context: Airflow automatically passes this. It contains:
                   - task_instance: Object to push/pull XCom data
                   - execution_date: When this DAG run started
                   - And more...
    """
    print(f"🔍 Checking PDF: {PDF_PATH}")
    
    # Check if file exists
    if not os.path.exists(PDF_PATH):
        # If not found, raise error (task will fail and show red in UI)
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    
    # Get file size in bytes
    file_size = os.path.getsize(PDF_PATH)
    
    # Print results (shows in Airflow logs)
    print(f"✅ PDF exists")
    print(f"📦 Size: {file_size / 1024:.2f} KB")  # Convert bytes to KB
    
    # ========================================================================
    # XCOM PUSH: Save data for other tasks
    # ========================================================================
    # This is like writing on a sticky note that other tasks can read
    
    context['task_instance'].xcom_push(
        key='file_size',    # Name of the "note" (you can call it anything)
        value=file_size     # What to write on the note (the number)
    )
    
    # Now other tasks can read this "file_size" note!
    print(f"💾 Saved file_size to XCom: {file_size}")

# ============================================================================
# TASK 2: CREATE SAMPLE CHUNKS
# ============================================================================

def create_sample_chunks(**context):
    """
    This function loads the PDF and creates 5 sample chunks.
    
    What it does:
    1. Reads file_size from Task 1 (XCom pull)
    2. Loads PDF and creates chunks
    3. Takes only first 5 chunks (for quick testing)
    4. Saves chunks to a file (using pickle)
    5. Saves metadata to XCom (for Task 3)
    """
    print(f"📝 Creating sample chunks...")
    
    # ========================================================================
    # XCOM PULL: Get data from previous task
    # ========================================================================
    # This is like reading the sticky note that Task 1 wrote
    
    file_size = context['task_instance'].xcom_pull(
        task_ids='check_pdf',   # Which task wrote the note
        key='file_size'          # Which note to read
    )
    
    # Now we have the file_size value from Task 1!
    print(f"📥 Got file_size from Task 1: {file_size / 1024:.2f} KB")
    
    # ========================================================================
    # Process PDF
    # ========================================================================
    
    print("📚 Loading PDF...")
    
    # Create Knowledge Base object
    kb = KnowledgeBase(
        pdf_path=PDF_PATH,
        chunk_size=200,      # 200 characters per chunk
        chunk_overlap=50     # 50 characters overlap between chunks
    )
    
    # Load PDF pages
    kb.load_pdf_data()
    
    # Create all chunks
    kb.create_chunks()
    
    # Take only first 5 chunks (for quick testing)
    sample_chunks = kb.chunks[:5]
    
    print(f"✅ Created {len(sample_chunks)} sample chunks")
    
    # ========================================================================
    # PICKLE: Save chunks to file
    # ========================================================================
    # Pickle = freeze Python objects to disk
    # Why? Because when this task ends, all variables disappear from memory
    # Pickle lets us save them to disk so Task 3 can load them
    
    import pickle
    
    # Where to save the pickle file
    chunks_file = os.path.join(OUTPUT_DIR, 'sample_chunks.pkl')
    
    # SAVE chunks to file
    with open(chunks_file, 'wb') as f:  # 'wb' = write binary mode
        pickle.dump(sample_chunks, f)    # Freeze chunks to file
    
    print(f"💾 Saved chunks to: {chunks_file}")
    
    # ========================================================================
    # Also save human-readable version (text file)
    # ========================================================================
    
    output_file = os.path.join(OUTPUT_DIR, 'sample_chunks.txt')
    
    with open(output_file, 'w') as f:  # Regular text file
        f.write(f"Sample Chunks from {PDF_PATH}\n")
        f.write("="*70 + "\n\n")
        
        # Write each chunk
        for i, chunk in enumerate(sample_chunks, 1):
            f.write(f"Chunk {i} (Page {chunk['page']}):\n")
            # Only show first 200 characters (preview)
            f.write(chunk['chunk'][:200] + "...\n\n")
    
    print(f"📄 Saved readable version to: {output_file}")
    
    # ========================================================================
    # XCOM PUSH: Save info for Task 3
    # ========================================================================
    
    context['task_instance'].xcom_push(
        key='num_chunks',
        value=len(sample_chunks)
    )
    
    context['task_instance'].xcom_push(
        key='output_file',
        value=output_file
    )
    
    context['task_instance'].xcom_push(
        key='chunks_file',
        value=chunks_file
    )
    
    print("✅ Task 2 complete!")

# ============================================================================
# TASK 3: PRINT SUMMARY
# ============================================================================

def print_summary(**context):
    """
    This function prints a summary of what the pipeline did.
    
    What it does:
    1. Reads data from Task 2 (XCom pull)
    2. Optionally loads chunks from pickle file (to demonstrate)
    3. Prints summary
    4. Saves summary to file
    """
    
    # ========================================================================
    # XCOM PULL: Get data from Task 2
    # ========================================================================
    
    num_chunks = context['task_instance'].xcom_pull(
        task_ids='create_sample_chunks',
        key='num_chunks'
    )
    
    output_file = context['task_instance'].xcom_pull(
        task_ids='create_sample_chunks',
        key='output_file'
    )
    
    chunks_file = context['task_instance'].xcom_pull(
        task_ids='create_sample_chunks',
        key='chunks_file'
    )
    
    print(f"📥 Got data from Task 2:")
    print(f"   - Number of chunks: {num_chunks}")
    print(f"   - Output file: {output_file}")
    print(f"   - Chunks pickle: {chunks_file}")
    
    # ========================================================================
    # OPTIONAL: Load chunks from pickle file (to demonstrate unpickling)
    # ========================================================================
    
    import pickle
    
    print(f"\n🔓 Loading chunks from pickle file...")
    
    # LOAD chunks from file
    with open(chunks_file, 'rb') as f:  # 'rb' = read binary mode
        chunks = pickle.load(f)          # Unfreeze chunks from file
    
    print(f"✅ Loaded {len(chunks)} chunks from pickle file")
    print(f"   First chunk preview: {chunks[0]['chunk'][:100]}...")
    
    # ========================================================================
    # Create summary
    # ========================================================================
    
    summary = f"""
    {"="*70}
    🎉 RAG PIPELINE TEST COMPLETED
    {"="*70}
    
    PDF File: {PDF_PATH}
    Sample Chunks Created: {num_chunks}
    
    Output Files:
    - Text file: {output_file}
    - Pickle file: {chunks_file}
    
    ✅ All tasks completed successfully!
    
    What we learned:
    1. Task 1: Checked PDF and saved file_size to XCom
    2. Task 2: Read file_size, created chunks, saved to pickle
    3. Task 3: Read metadata, loaded pickle file, printed summary
    
    Next steps:
    1. Look at the text file to see your chunks
    2. Try adding more tasks (embeddings, indexing)
    3. Build the full pipeline!
    
    {"="*70}
    """
    
    print(summary)
    
    # Save summary to file
    summary_file = os.path.join(OUTPUT_DIR, 'pipeline_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"📄 Summary saved to: {summary_file}")
    
    print("✅ Task 3 complete!")

# ============================================================================
# STEP 3: Create Task Objects
# ============================================================================
# These are the boxes you see in the Airflow UI

task1 = PythonOperator(
    task_id='check_pdf',           # ID shown in UI (must be unique)
    python_callable=check_pdf,     # Which function to run
    dag=dag,                       # Which DAG this task belongs to
)

task2 = PythonOperator(
    task_id='create_sample_chunks',
    python_callable=create_sample_chunks,
    dag=dag,
)

task3 = PythonOperator(
    task_id='print_summary',
    python_callable=print_summary,
    dag=dag,
)

# ============================================================================
# STEP 4: Define Task Order (Dependencies)
# ============================================================================
# The >> operator means "then"
# Read as: "task1 THEN task2 THEN task3"

task1 >> task2 >> task3

# KNOWLEDGE PURPOSES ONLy
# Could also write as:
# task1.set_downstream(task2)
# task2.set_downstream(task3)

# Or:
# task3.set_upstream(task2)
# task2.set_upstream(task1)

# ============================================================================
# END OF DAG
# ============================================================================

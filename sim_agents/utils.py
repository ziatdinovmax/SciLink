import os
import logging
from datetime import datetime



def save_generated_script(script_content: str, description: str, attempt: int, output_dir: str) -> str | None:
    """Saves the script content to a file and returns the path."""
    try:
        # Ensure directory exists (might be better done once on agent init)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for use in filename
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30]).rstrip("_")
        filename = f"ase_script_{safe_desc}_attempt{attempt}_{timestamp}.py"
        saved_script_path = os.path.join(output_dir, filename)

        with open(saved_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        logging.info(f"Saved script for attempt {attempt} to: {saved_script_path}")
        return saved_script_path
    except IOError as e:
        logging.error(f"Failed to save script for attempt {attempt}: {e}")
        return None
    except Exception as e: # Catch broader exceptions during save
        logging.error(f"Unexpected error saving script for attempt {attempt}: {e}")
        return None
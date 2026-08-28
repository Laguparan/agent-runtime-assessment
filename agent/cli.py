import argparse
import uuid
import sys
from loop import run_agent

def main():
    parser = argparse.ArgumentParser(description="Adversarial Agent Runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. 'run' command
    run_parser = subparsers.add_parser("run", help="Start a new autonomous agent run")
    run_parser.add_argument("--task", required=True, help="The goal for the agent to achieve")

    # 2. 'resume' command (R2 requirement)
    resume_parser = subparsers.add_parser("resume", help="Resume a killed run from SQLite")
    resume_parser.add_argument("run_id", help="The ID of the run to resume")

    # 3. 'replay' command (R6 requirement)
    replay_parser = subparsers.add_parser("replay", help="Replay a trace without network calls")
    replay_parser.add_argument("run_id", help="The ID of the run to replay")

    args = parser.parse_args()

    if args.command == "run":
        # Generate a unique ID for this run
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        try:
            run_agent(run_id, args.task)
        except KeyboardInterrupt:
            print(f"\n[!] Process manually interrupted. Use 'python agent/cli.py resume {run_id}' to recover.")
            sys.exit(1)
            
    elif args.command == "resume":
        # Note: To fully implement R2, this would fetch the last state from storage.py
        # and pass the history back into run_agent(). 
        print(f"🔧 [Stub] Resuming run {args.run_id} from SQLite state...")
        
    elif args.command == "replay":
        # Note: To fully implement R6, this would parse your .jsonl trace file
        # and print the conversation history without calling the mock_client.
        print(f"⏪ [Stub] Replaying trace for {args.run_id}...")

if __name__ == "__main__":
    main()
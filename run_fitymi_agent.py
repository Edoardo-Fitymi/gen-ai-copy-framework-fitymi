import argparse
import asyncio
import sys
import json
from pathlib import Path
from nexus import FitymiNexus, NexusContext


def read_brief_file(brief_path: str) -> str:
    """Read brief content from a file path.
    
    Args:
        brief_path: Path to the brief file (relative or absolute)
        
    Returns:
        The content of the brief file
        
    Raises:
        FileNotFoundError: If the brief file does not exist
    """
    path = Path(brief_path)
    
    # Try relative path first, then absolute
    if not path.is_absolute():
        # Check relative to current working directory
        if not path.exists():
            # Try relative to script location
            script_dir = Path(__file__).parent
            path = script_dir / brief_path
    
    if not path.exists():
        raise FileNotFoundError(f"Brief file not found: {brief_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


async def main():
    parser.add_argument("--task", type=str, required=True, help="Es: 'Landing Page B2B'")
    parser.add_argument("--brand", type=str, default="Unspecified Brand", help="Brand name")
    parser.add_argument("--audience", type=str, default="General Audience", help="Target audience")
    parser.add_argument("--product", type=str, default="Unspecified Product", help="Product being advertised")
    parser.add_argument("--goal", type=str, default="Conversion", help="Main goal of the copy")
    parser.add_argument("--brief", type=str, default=None, help="Brief content as a string")
    parser.add_argument("--brief_path", type=str, default=None, help="Path to a brief file (takes precedence over --brief)")
    args = parser.parse_args()

    # Determine brief content
    brief_content = ""
    if args.brief_path:
        try:
            brief_content = read_brief_file(args.brief_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.brief:
        brief_content = args.brief

    print("🚀 Initializing Fitymi Nexus Swarm Intelligence...")
    nexus = FitymiNexus()
    
    context = NexusContext(
        brand=args.brand,
        target_audience=args.audience,
        product=args.product,
        goal=args.goal,
        task_type=args.task,
        constraints={"brief_content": brief_content, "tone": "human-first, assertivo, zero hype"}
    )

    result = await nexus.execute_workflow(context)
    
    print("\n" + "="*50)
    print("🌟 FITYMI NEXUS: FINAL COGNITIVE SWARM COPY 🌟")
    print("="*50 + "\n")
    print(result.get("final_copy", result))
    
    print("\n" + "="*50)
    print(f"🏆 Final Score from autonomous evaluation: {result.get('final_score', 'N/A')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

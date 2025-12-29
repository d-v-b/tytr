"""Command-line interface for tytr type transformations."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import get_type_hints


def load_class_from_path(file_path: str, class_name: str) -> type:
    """
    Load a class from a Python file.

    Parameters
    ----------
    file_path : str
        Path to the Python file
    class_name : str
        Name of the class to load

    Returns
    -------
    type
        The loaded class

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    AttributeError
        If the class is not found in the module
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("_tytr_target_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_tytr_target_module"] = module
    spec.loader.exec_module(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in {file_path}. "
            f"Available: {', '.join(dir(module))}"
        )

    return getattr(module, class_name)


def format_type_annotation(annotation: type) -> str:
    """
    Format a type annotation as a string for code generation.

    Parameters
    ----------
    annotation : type
        The type annotation to format

    Returns
    -------
    str
        String representation suitable for code generation
    """
    # Handle basic types
    if hasattr(annotation, "__module__") and hasattr(annotation, "__name__"):
        module = annotation.__module__
        name = annotation.__name__

        # Built-in types
        if module == "builtins":
            return name

        # typing module types - use simple names
        if module == "typing":
            import re
            repr_str = repr(annotation).replace("typing.", "")
            # Convert Union[X, Y] to X | Y for modern syntax
            union_pattern = r'Union\[([^\]]+)\]'
            if re.search(union_pattern, repr_str):
                def replace_union(match):
                    union_args = match.group(1)
                    return union_args.replace(', ', ' | ')
                repr_str = re.sub(union_pattern, replace_union, repr_str)
            return repr_str

        # typing_extensions
        if module == "typing_extensions":
            return repr(annotation).replace("typing_extensions.", "")

        # Types from the loaded user module - use just the class name
        if module == "_tytr_target_module":
            return name

        # Other types - use qualified name
        return f"{module}.{name}"

    # For generic types, use repr and clean it up
    repr_str = repr(annotation)
    # Remove module prefixes for common typing constructs
    repr_str = repr_str.replace("typing.", "")
    repr_str = repr_str.replace("typing_extensions.", "")
    # Remove _tytr_target_module prefix from user's types
    repr_str = repr_str.replace("_tytr_target_module.", "")

    # Convert Union[X, Y] to X | Y for modern syntax
    import re
    union_pattern = r'Union\[([^\]]+)\]'
    if re.search(union_pattern, repr_str):
        # Extract the types inside Union[...] and convert to pipe syntax
        def replace_union(match):
            union_args = match.group(1)
            return union_args.replace(', ', ' | ')
        repr_str = re.sub(union_pattern, replace_union, repr_str)

    return repr_str


def generate_code(
    cls: type,
    transformation: str,
    name: str | None = None,
    key_delimiter: str | None = None,
) -> str:
    """
    Generate Python source code for type transformations.

    Parameters
    ----------
    cls : type
        The source class to transform
    transformation : str
        The transformation to apply
    name : str, optional
        Custom name for the generated type
    key_delimiter : str, optional
        Key delimiter for flatten transformation (default: "_")

    Returns
    -------
    str
        Python source code for the transformation
    """
    from tytr.core import flatten, key_of, to_typeddict, value_of
    from tytr.utilities import partial, readonly, required

    imports: set[str] = set()
    imports.add("from __future__ import annotations")
    body_lines: list[str] = []

    # Handle type aliases (key_of, value_of)
    if transformation in ("key_of", "value_of"):
        td = to_typeddict(cls)
        if transformation == "key_of":
            result_type = key_of(td)
            default_name = f"{cls.__name__}Keys"
        else:  # value_of
            result_type = value_of(td)
            default_name = f"{cls.__name__}Values"

        alias_name = name if name is not None else default_name
        type_str = format_type_annotation(result_type)

        if "Literal[" in type_str:
            imports.add("from typing import Literal")

        body_lines.append(f"{alias_name} = {type_str}")

    # Handle protocols (getter, setter)
    elif transformation in ("getter", "setter"):
        td = to_typeddict(cls)
        hints = get_type_hints(td, include_extras=True)

        if name is None:
            func_name = "gets" if transformation == "getter" else "sets"
            protocol_name = f"{func_name.title()}Protocol"
        else:
            protocol_name = name

        imports.add("from typing import Literal, Protocol, overload")

        body_lines.append("")
        body_lines.append(f"class {protocol_name}(Protocol):")

        for field_name, field_type in hints.items():
            type_str = format_type_annotation(field_type)
            body_lines.append("    @overload")
            if transformation == "getter":
                body_lines.append(
                    f'    def __call__(self, key: Literal["{field_name}"]) -> '
                    f"{type_str}: ..."
                )
            else:  # setter
                body_lines.append(
                    f'    def __call__(self, key: Literal["{field_name}"], '
                    f"value: {type_str}) -> None: ..."
                )
            body_lines.append("")

    # Handle TypedDict transformations
    else:
        if transformation == "typeddict":
            result_type = to_typeddict(cls, name=name)
        elif transformation == "flatten":
            result_type = flatten(cls, name=name, key_delimiter=key_delimiter)
        elif transformation == "partial":
            base_td = to_typeddict(cls)
            result_type = partial(base_td, name=name)
        elif transformation == "required":
            base_td = to_typeddict(cls)
            result_type = required(base_td, name=name)
        elif transformation == "readonly":
            base_td = to_typeddict(cls)
            result_type = readonly(base_td, name=name)
        else:
            raise ValueError(
                f"Unknown transformation: {transformation}. "
                f"Valid options: typeddict, key_of, value_of, getter, setter, "
                f"flatten, partial, required, readonly"
            )

        # Get the annotations
        hints = get_type_hints(result_type, include_extras=True)

        # Collect required imports
        imports.add("from typing_extensions import TypedDict")

        # Check what typing constructs we need
        typing_imports = []
        needs_readonly = False

        for field_type in hints.values():
            type_str = format_type_annotation(field_type)
            if "Required[" in type_str and "Required" not in typing_imports:
                typing_imports.append("Required")
            if "NotRequired[" in type_str and "NotRequired" not in typing_imports:
                typing_imports.append("NotRequired")
            if "ReadOnly[" in type_str:
                needs_readonly = True

        if typing_imports:
            imports.add(f"from typing import {', '.join(sorted(typing_imports))}")

        if needs_readonly:
            imports.add("from typing_extensions import ReadOnly")

        # Generate TypedDict class
        body_lines.append(f"class {result_type.__name__}(TypedDict):")
        if not hints:
            body_lines.append("    pass")
        else:
            for field_name, field_type in hints.items():
                type_str = format_type_annotation(field_type)
                body_lines.append(f"    {field_name}: {type_str}")

    # Assemble final output
    lines: list[str] = []
    lines.append(f"# Generated by tytr gen {transformation}")
    lines.extend(sorted(imports))
    lines.append("")
    lines.extend(body_lines)
    lines.append("")

    return "\n".join(lines)


def cmd_gen(args: argparse.Namespace) -> int:
    """
    Execute the 'gen' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse the target (file_path::class_name)
        if "::" not in args.target:
            print(
                "Error: Target must be in format 'path/to/file.py::ClassName'",
                file=sys.stderr,
            )
            return 1

        file_path, class_name = args.target.split("::", 1)

        # Load the class
        cls = load_class_from_path(file_path, class_name)

        # Generate the code
        code = generate_code(
            cls, args.transformation, args.name, args.key_delimiter
        )

        # Output the code
        print(code)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point for the tytr CLI.

    Returns
    -------
    int
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="tytr",
        description="Generate TypedDict transformations from Python classes",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # gen command
    gen_parser = subparsers.add_parser(
        "gen", help="Generate TypedDict code from a class"
    )
    gen_parser.add_argument(
        "transformation",
        choices=[
            "typeddict",
            "key_of",
            "value_of",
            "getter",
            "setter",
            "flatten",
            "partial",
            "required",
            "readonly",
        ],
        help="Type transformation to apply",
    )
    gen_parser.add_argument(
        "target",
        help="Target class in format 'path/to/file.py::ClassName'",
    )
    gen_parser.add_argument(
        "--name",
        "-n",
        help="Custom name for the generated TypedDict",
    )
    gen_parser.add_argument(
        "--key-delimiter",
        "-d",
        help="Key delimiter for flatten transformation (default: '_')",
    )
    gen_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose error output",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "gen":
        return cmd_gen(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())

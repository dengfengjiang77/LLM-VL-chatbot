import importlib
import llama_index

def print_package_contents(package, parent_name=""):
    for attr_name in dir(package):
        if not attr_name.startswith("_"):
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            try:
                attr = getattr(package, attr_name)
                if isinstance(attr, type(package)):
                    print(f"{full_name}")
                    print_package_contents(attr, full_name)
            except ImportError:
                continue

print_package_contents(llama_index)

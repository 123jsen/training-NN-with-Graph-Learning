def read_designs(path):
    """Reads array of designs from given path"""

    designs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            design = []

            line = line.rstrip("\n")
            line = line.split(', ')

            for num in line:
                try:
                    if num:
                        design.append(int(num))
                except ValueError:
                    print(f"Warning: Cannot append [{num}]")

            designs.append(design)

    return designs

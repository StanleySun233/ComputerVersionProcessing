class signal:
    def __init__(self, left, right):
        self.left = left
        self.right = right


def solve(s: str, area: dict):
    left = area[s[0]].left
    right = area[s[0]].right

    for i in s[1:]:
        temp = left
        left += (right - left) * area[i].left
        right = temp + (right - temp) * area[i].right

    return (left + right) / 2


if __name__ == "__main__":
    A = signal(0, 0.5)
    B = signal(0.5, 0.7)
    C = signal(0.7, 1)
    sheet = {"A": A, "B": B, "C": C}
    result = solve(input(), sheet)
    print(result)

# What the model actually says

All 34 prompts from [`prompts.tsv`](prompts.tsv), run through `clover-k3`,
showing the 12 tokens it generated for each.

The prompt is on the first line. Everything after `->` is the model's own
output, shown exactly as it came out, including line breaks.

---

### 1

```
The capital of France is
-> Paris.",
+            "The Eiffel Tower is located in
```

### 2

```
The chemical symbol for gold is
-> Au, which comes from the Latin word "aurum,"
```

### 3

```
The largest planet in our solar system is
-> Jupiter. It is a gas giant with a diameter of about
```

### 4

```
Water boils at a temperature of
-> 212°F, which is the boiling point of water at
```

### 5

```
The author of Pride and Prejudice was
-> Jane Austen. It was first published in 1813
```

### 6

```
In 1969 the first humans landed on the
-> moon. The Apollo 11 mission was a monumental achievement in
```

### 7

```
The speed of light in a vacuum is approximately
-> 299,792,458 meters per second. This speed
```

### 8

```
The mitochondria is often called the powerhouse of the
-> cell because it produces energy in the form of ATP through cellular
```

### 9

```
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(
-> n-2)
print(fibonacci(10)) #
```

### 10

```
SELECT name, COUNT(*) FROM users WHERE active = 1 GROUP BY
-> name HAVING COUNT(*) > 1;
+```
+

```

### 11

```
public static void main(String[] args) { System.out.println(
->“Hello World”); } } 编译并运行上面的
```

### 12

```
for (int i = 0; i < n; i++) { total += arr[i]; } return
-> total; } int main() { int arr[] = {
```

### 13

```
git commit -m "fix: handle null pointer in the
-> user service"
```

### 3. Push to Remote


```

### 14

```
The derivative of x squared with respect to x is
-> 2x. The derivative of 2x with respect
```

### 15

```
If 3x plus 7 equals 22 then x equals
-> 5. To solve for x, subtract 7 from
```

### 16

```
The sum of the interior angles of a triangle is
-> 180 degrees. This is a fundamental property of triangles and
```

### 17

```
Seven multiplied by eight equals fifty
->-six.

Seven multiplied by nine equals sixty-three.

Seven multiplied
```

### 18

```
yes yes yes yes yes yes yes yes yes yes yes yes
-> yes yes yes yes yes yes yes yes yes yes yes yes
```

### 19

```
the the the the the the the the the the the the
-> the the the the the the the the the the the the
```

### 20

```
one two three one two three one two three one two
-> three one two three one two three one two three one two
```

### 21

```
a a a a b b b b a a a a b b b b a a a a b b b b
-> a a a a b b b b a a a a
```

### 22

```
Paris is the capital of France and Berlin is the capital of
-> Germany. Paris is the capital of France and Berlin is the
```

### 23

```
Paris is the capital of France and Rome is the capital of
-> Italy. Both cities are known for their rich history, art
```

### 24

```
Paris is the capital of France, a city on the river Seine that has long
-> been a center of art, culture, and fashion. It
```

### 25

```
Paris is the capital of France, a city that sits far inland away from any
-> ocean. The Seine River flows through Paris, but it is
```

### 26

```
She opened the door slowly, listening for any sound in the hallway beyond, but heard
-> nothing. She stepped out into the corridor, her bare feet
```

### 27

```
The committee reviewed the proposal for three hours before deciding that further work was
-> needed. The committee will meet again on May 15 to
```

### 28

```
Despite the heavy rain that had fallen throughout the night, the river remained below its
-> banks. The water was high, but not dangerously so.
```

### 29

```
Machine learning models trained on large corpora of text tend to exhibit behaviour that
-> is consistent with the biases present in the training data. This
```

### 30

```
The quick brown fox jumps over the lazy dog while the cat watches from the
-> window. The sun shines brightly in the clear blue sky as
```

### 31

```
Once upon a time in a village at the foot of a great mountain there lived a
-> young man named Taro. He was a woodcutter,
```

### 32

```
La capitale de la France est
-> Paris.",
+            "La France est un pays d'
```

### 33

```
Die Hauptstadt von Frankreich ist
-> Paris.",
+            "Die Hauptstadt von Deutschland ist Berlin
```

### 34

```
El idioma oficial de Mexico es
-> el espanol, pero tambien se hablan much
```

---

## How this was produced

One rented CPU machine, no GPU: Ryzen 9 7950X3D, 16 cores, 124 GB RAM,
NVMe RAID1, holding the released Kimi K3 checkpoint at 1.5 TB.

`clover-k3` is one C file that runs the model from its equation. It computes
one token per pass, so [`gen.py`](gen.py) calls it in a loop, carrying the
attention cache and recurrent state forward so each new word costs one
position rather than the whole prompt again.

```
34 prompts, 12 tokens each
2526 s in total   74.3 s per prompt   6.19 s per token
```

```sh
./build.sh
./gate.sh    # must print md5 23d162dcefb18211a7540ef12948f1eb
python3 gen.py --ids 1008,10484,318,15383,387 12 prefix
```

**Two things worth knowing when reading the output.** These prompts are bare
completions with no chat template, so the model continues them as if they
were lines in a file rather than questions to answer; that is why a few run
on into quotes, diff markers or a second language. And six seconds a word is
not interactive — a paragraph takes minutes.

The side-by-side correctness comparison against an independent implementation
of the same model — 34 of 34 identical answers, 3.40x less wall time — is in
[`clover-k3-comparison.md`](clover-k3-comparison.md), and the measurement
record in
[`../k3-analysis/clover-scaling-architecture.md`](../k3-analysis/clover-scaling-architecture.md).

; gcd(a, b) by repeated subtraction (a, b > 0); result in a
let a = 1
let b = 2
let t = 3
loop:
  eq  t, a, b
  jnz t, done
  lt  t, a, b        ; t = (a < b)
  jnz t, swap
  sub a, a, b        ; a > b:  a -= b
  jmp loop
swap:
  sub b, b, a        ; a < b:  b -= a
  jmp loop
done:
  halt

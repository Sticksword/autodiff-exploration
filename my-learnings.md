# Automatic differentiation exploration

## Steps taken

1. [Ran initial google search, came across this article that was superb intro](https://medium.com/@marksaroufim/automatic-differentiation-step-by-step-24240f97a6e6)

2. [This was the first tutorial that was relatively good](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)

3. I found this ["Rudimentary automatic differentiation framework" implementation](https://github.com/bgavran/autodiff) but I wasn't able to understand it at first. So I went and searched for a more simple implementation.

4. [And I found one! I read the simple python source code and got pretty good idea of reverse mode AD or AD for short](https://github.com/Rufflewind/revad/blob/master/revad.py)

5. At this point I have a pretty decent grasp of numerical, symbolic, and automatic differentiation. I figured since Tensorflow and all the other famous deep learning frameworks use AD, I would try and learn it too. I had heard of it but didn't really know what it was. I was excited.

6. My summary of the approaches was nicely summed up by the first Medium article I linked.

   ``` text
   Forward Mode AD can throw away intermediate results since it’s an iterative algorithm.
   Reverse Mode AD needs to keep all intermediate results in memory since it’s a recursive algorithm.
   Forward Mode AD needs to run once per input to compute the full Jacobian matrix.
   Reverse Mode AD needs to run once per output to compute the full Jacobian matrix.
   ```

   tl;dr many inputs means forward mode is slow, many outputs means backwards mode is slow

   backwards AD also takes up a lot of memory as I experienced from setting n = 16 (18 crashed zsh terminal)

7. I was still not satisfied with my understanding. I felt like I wanted further validation, perhaps further implementation examples in order to solidify my understanding. I was lucky enough to stumble upon [this course taught by U Washington](http://dlsys.cs.washington.edu/) and I was instantly hooked. I read through lectures [3](http://dlsys.cs.washington.edu/pdf/lecture3.pdf) and [4](http://dlsys.cs.washington.edu/pdf/lecture4.pdf) and now I feel a lot better about my understanding but it still wasn't enough. I wanted to see a more complex code example.

8. Enter the course's [first assignment](https://github.com/dlsys-course/assignment1). I basically did the assignment over the weekend, trying to grok the concept of AD through code. You don't really understand something until you either teach it or implement it in code.

9. I wrangled the expected API from the given assignment and molded it to fit the Autodiff framework assignment 1 had provided me.

## File descriptions

* alt-simple-AD.py - a very simple implementation of AD that first helped me grok what AD is
* autodiff.py - the UW assignment 1 that basically creates an autodiff framework
* autodiff_test.py - tests for assignment 1
* anyscale-simple-AD.py - my code that implements the API that was outlined in the takehome assignment

### Getting started / running code

``` bash
pipenv install
pipenv run python anyscale-simple-AD.py
```

## Summary

Overall very fun exploration, learned more than I expected about deep learning frameworks and the fundamentals of deep learning. I would say my understanding _still_ feels fuzzy sometimes but overall I get what AD is and how it impacts deep learning and neural nets.

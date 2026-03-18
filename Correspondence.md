Feda Curic   10:36 AM
Hi, I've been reading the paper on PSVI you recently posted.
Have you tried using a sparse precision matrix rather than a sparse covariance?

A sparse precision matrix naturally accommodates long-range marginal correlations by encoding conditional independence. In contrast, forcing sparsity on the covariance matrix imposes marginal independence, which risks suppressing valid long-range dependencies.

I quickly hacked together an implementation of PSVI using the precision matrix parameterization and ran it on a simple 2D heat-equation example. The results look promising: the precision-based PSVI seems to yield a larger spread that better matches the uncertainty of the observations.

My implementation is wonky and probably has bugs, but I think the idea of using a sparse precision matrix is reasonable.
Wanted to share in case you find it useful for future work.

Kind regards,
Feda

Click or press enter to display in the image preview
ES in the plot is the vanilla Ensemble Smoother by the way.

Andrew Curtis sent the following message at 10:59 AM
View Andrew’s profileAndrew Curtis
Andrew Curtis   10:59 AM
Interesting idea Feda, many thanks for sending!
 
When you say that the precision uncertainty better matches the uncertainty in observations, I am not quite seeing what you mean: perhaps I don't understnad what the red errors are (+/- 1 standard deviation?) and how you have coloured the yellow zone (spans every realisation up to (e.g.) +/- 3 standard deviations?)
 
All the best,
Andrew.

Feda Curic sent the following message at 11:07 AM
View Feda’s profileFeda Curic
Feda Curic   11:07 AM
The red dots and error-bars are my uncertain observations, and the error bars represent that uncertainty.

The gray lines are the responses I get when running the heat equation using the prior parameter field. 

The field represents conductivities and the responses represent temperatures.

The yellow lines are the responses I get when running the heat equation using the posterior parameter field.

PSVI with covariance seems to produce resposes that are quite close to the observations, while PSVI with precision has more varying resposes.

Does this make sense?(Edited)

Andrew Curtis sent the following messages at 2:57 PM
View Andrew’s profileAndrew Curtis
Andrew Curtis   2:57 PM
Yes, perfect sense - thank you. 

However, it raises another question: why do you expect (want) the yellow results to deviate so far from your measurement uncertainties, as you get with the precision formulation? It seems to me that the covariance formulation honours your data uncertainties better - or am I misunderstanding?

View Andrew’s profileAndrew Curtis
Andrew Curtis   3:00 PM
PS Also, I'm not sure what the problem set up is (where the data are, what your prior distribution on parameters is) but how do the posterior distributions in parameter space look for each case? 
 
PPS I really appreciate you looking into PSVI so rapidly - I'm impressed!

Feda Curic sent the following message at 3:25 PM
View Feda’s profileFeda Curic
Feda Curic   3:25 PM
In history matching using ES-MDA we aim for a posterior that produces responses that vary according to the errors in observations. We don't want a posterior with very low variance that is almost deterministic.
Unless our observations are highly accurate that is, which they never are.
There's so much uncertainty in the measurements we use, that we can't really get a posterior with low variance unless we overfit.
Hope that makes some sense.

Here's the exact problem set-up and code I use:
https://github.com/equinor/dass/blob/main/notebooks/ES_2D_Heat_Equation.ipynb

This does not include my implementation of PSVI yet because it is not ready for sharing yet.
I'll send you a link once I have it ready if you are interested.

I've been wanting to learn a bit about variational methods for some time, and your paper seemed like a good place to start, so thanks!


dass/notebooks/ES_2D_Heat_Equation.ipynb at main · equinor/dass
github.com
Andrew Curtis sent the following message at 3:54 PM
View Andrew’s profileAndrew Curtis
Andrew Curtis   3:54 PM
Thanks a lot Feda. If you don't mind, I'll ask my PhD student (the first author) to take a good look at this code because I'm booked upwith teaching etc. right now, and then get back to you? 
 
 I generally agree with you about not wanting to over-fit the data. However, it's just that when I look at your plots, the red errors bars on the data are very small, so you as far as the inverse problem is concerned, you haven't modelled the data as being particularly uncertain - actually you have defined them as fairly accurate. That is why PSVI with covariances is fitting the data so closely; if you want the posterior to provide more uncertainty in data space, then the uncertainties on the data should be larger. 
 
But again, this argument depends on exactly what you are plotting with the red error bars: you said they are "uncertainties", but do you mean +/-1 standard deviation of a Gaussian measurement uncertainty, or do you mean those are the bounds of uniform distributions so that the red error bars define absolute bounds on the data measurement accuracies? If they are uniform distributions then the PSVI solution using covariances is producing the better result because it conforms to your data distribution, while the precision matrix formulation does not (in that case). If they are Gaussians then ther reverse may be true.

Feda Curic sent the following message at 4:06 PM
View Feda’s profileFeda Curic
Feda Curic   4:06 PM
The uncertainties are modeled as gaussians. I think that I ran a whole bunch of iterations to produce that plot and the fit is less good if I run fewer iterations. More testing is definitely needed.

Your student can feel free to reach out to me if she/he has any questions.

Andrew Curtis sent the following messages at 4:29 PM
View Andrew’s profileAndrew Curtis
Andrew Curtis   4:29 PM
👏
👍
😊



Great, thank you.
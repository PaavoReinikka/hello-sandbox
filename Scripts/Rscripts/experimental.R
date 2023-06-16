k <- 100000
n <- 3
FC <- 1.5
q_st <- qnorm(0.9,0,1)
q_st
# we want q=FC/2 => kerroin=FC/2q_st 
sigma <- FC/(2*q_st)
#sigma <- 1
r <- c(rnorm(k,0,sigma),rnorm(k,-FC/2,sigma),rnorm(k,FC/2,sigma))
r <- sample(r,10000,TRUE)
var(r)
sigma^2 + FC^2/(2*n)


sigma^2 + FC^2*(n-1)/n^2

sigma^2 + FC^2*(n-2)/n^2


hist(r)



pA=0.3
pB=0.7
mA=3
mB=-2

pA*mA^2 + pB*mB^2 - (pA*mA + pB*mB)^2
pA*pB*(mA-mB)^2

#############################################################################
library(extraDistr)
library('comprehenr')

n <- 20
sig1 <- 1
sig2 <- 2

effect_range <- seq(0,5,0.25)
results <- data.frame(matrix(ncol = 6,nrow = 0))

for(effect in effect_range) {
  MLeffects <- c()
  MLhighs <- c()
  MLlows <- c()
  Bayeseffects <- c()
  Bayeshighs <- c()
  Bayeslows <- c()
  MLerrors <- c()
  Berrors <- c()
  for(i in 1:10) {
    # data
    r1 <- rnorm(n, 0, sig1)
    r2 <- rnorm(n, effect, sig2)
    
    # ML estimates
    m1 <- mean(r1)
    m2 <- mean(r2)
    std <- (var(r1)/n + var(r2)/n) %>% sqrt()
    ML_effect <- m2 - m1
    ML_high <- ML_effect + 1.96*std
    ML_low <- ML_effect - 1.96*std
    
    # Posterior estimates
    sigma1 <- rinvchisq(k,n-1,sqrt(var(r1)))/n %>% sqrt()
    sigma2 <- rinvchisq(k,n-1,sqrt(var(r2)))/n %>% sqrt()
    mu1 <- to_vec(for(i in 1:length(sigma1)) rnorm(1, m1, sigma1[i]))
    mu2 <- to_vec(for(i in 1:length(sigma2)) rnorm(1, m2, sigma1[i]))
    posterior_effect <- (mu2 - mu1)
    Bayes_effect <- posterior_effect %>% mean()
    tmp <- quantile(posterior_effect,probs = c(0.025, 0.975))
    Bayes_high <- tmp[2]
    Bayes_low <- tmp[1]
    
    MLeffects <- c(MLeffects,ML_effect)
    MLhighs <- c(MLhighs,ML_high)
    MLlows <- c(MLlows,ML_low)
    Bayeseffects <- c(Bayeseffects,Bayes_effect)
    Bayeshighs <- c(Bayeshighs,Bayes_high)
    Bayeslows <- c(Bayeslows,Bayes_low)
    MLerrors <- c(MLerrors,abs(ML_effect-effect))
    Berrors <- c(MLerrors,abs(Bayes_effect-effect))
    
  }
  # insert the results in the data frame
  #results <- rbind(results, c(ML_effect,ML_high,ML_low,Bayes_effect,Bayes_high,Bayes_low))
  results <- rbind(results, 
                   c(mean(MLeffects),
                     mean(MLhighs),
                     mean(MLlows),
                     mean(MLerrors),
                     mean(Bayeseffects),
                     mean(Bayeshighs),
                     mean(Bayeslows),
                     mean(Berrors)))
  
}

results <- cbind(results, effect_range)
colnames(results) <- c('MLest','MLhigh','MLlow','MLerror','Best','Bhigh','Blow','Berror','true_effect')


ggplot(results, aes(x=true_effect)) + 
  geom_line(aes(y = true_effect)) + 
  geom_line(aes(y = MLest), color='red') +
  geom_line(aes(y = MLhigh), lty='dashed', color='red') +
  geom_line(aes(y = MLlow), lty='dashed', color='red') +
  geom_line(aes(y = Best),color='blue') +
  geom_line(aes(y = Bhigh), lty='dashed', color='blue') +
  geom_line(aes(y = Blow), lty='dashed', color='blue')

ggplot(results, aes(x=true_effect)) + 
  geom_line(aes(y = MLerror, color='red')) +
  geom_line(aes(y = Berror, color='blue', lty='dashed'))
  
##########################################

hist(r2-r1)
abline(v=ML_effect, col='red', lwd=3)
abline(v=ML_high, col='black', lwd=3)
abline(v=ML_low, col='black', lwd=3)

hist(posterior_effect)
abline(v=Bayes_effect, col='blue', lwd=3)
abline(v=quantile(posterior_effect,probs = c(0.159, 1-0.159)), lwd=3)

hist(r2-r1)
abline(v=ML_effect, col='red', lwd=3)
abline(v=Bayes_effect, col='blue', lwd=3, lty=3)
abline(v=c(ML_high,ML_low), col='yellow', lwd=2)
abline(v=quantile(posterior_effect,probs = c(0.159, 1-0.159)), col='green', lwd=2, lty=2)



#################################################################
################# check/set the environment #####################

#getwd()
#setwd("../../../../hello-sandbox/Scripts/plasma/")
#setwd('../Rscripts/')

library(tidyverse)
library(brms)

############### LOAD AND CHECK THE DATA #########################

# filenum (from 1 to 10)
fnum <- 1
path <- "../../../PARKINSONS/PDproj/plasmadata/clean/"
filenames <- readLines(paste(path, 'filenames.txt', sep = ''))
fname <- filenames[fnum]
fname_full <- paste(path, fname, sep='')

data <- read.csv2(fname_full, header = F)

gender <- data[1,9:80] %>% factor()
group <- data[2,9:80] %>% factor()
area <- data[-c(1:4),9:80] %>% sapply(., as.numeric) %>% as.matrix()

###############################################################################
# Make full design matrix (now all features included -- select for each model)
# First column corresponds with male(1)/female(0),
# second with pd(1)/control(0)
# third with gender/case interaction, i.e., male*pd
design <- gender %>% as.numeric() %>% -1 %>%
  cbind(., group %>% as.numeric() %>% -1) %>% data.frame()
colnames(design) <- c('male','pd')
design$male_pd <- design$male * design$pd
design <- design %>% cbind(., area %>% t())
colnames(design) <- c(colnames(design)[1:3],paste0("A", 1:(ncol(design)-3)))
##############################################################################

model <- brm(A1 ~ pd + male + male_pd, data=design, prior = set_prior("normal(20,10)"), class='b')
summary(model)
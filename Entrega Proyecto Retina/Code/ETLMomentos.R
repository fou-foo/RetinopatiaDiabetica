library(readxl)
setwd("~/Desktop/RetinopatiaDiabetica/Code/New")
files <- dir()[2:5]
imagenes.nombres <- read.csv(files[1], header = FALSE)
for(i in 2:length(files))
{
    archivo <- read.csv( files[i], header = FALSE)
    imagenes.nombres <- rbind(imagenes.nombres, archivo)
}
library(dplyr)
names(imagenes.nombres)
imagenes.nombres %>% group_by(V1) %>% mutate( y = V2 >0   ) -> imagenes.nombres
class(imagenes.nombres) <- 'data.frame'
#imagenes.nombres$y <- as.numeric(imagenes.nombres$y)
write.csv(imagenes.nombres, file='momentos.csv')

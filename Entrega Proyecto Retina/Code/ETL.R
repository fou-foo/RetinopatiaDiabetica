library(readxl)
setwd("~/Desktop/CD2EDUARDOFOO/Retina/Code")
files <- dir()
imagenes.nombres <- read_excel(path = files[1])
for(i in 2:length(files))
{
    archivo <- read_excel(path = files[i])
    imagenes.nombres <- rbind(imagenes.nombres, archivo)
}
library(dplyr)
names(imagenes.nombres)
imagenes.nombres %>% group_by(`Image name`) %>% mutate( y = `Retinopathy grade` >0   ) -> imagenes.nombres
class(imagenes.nombres) <- 'data.frame'
imagenes.nombres$y <- as.numeric(imagenes.nombres$y)
write.csv(imagenes.nombres, file='metadata.csv', row.names = FALSE)

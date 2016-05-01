#=================================================================================================
# Computational Cancer Biology Analysis (CCBA) Library
# Pablo Tamayo Dec 30, 2015
#
#-------------------------------------------------------------------------------------------------

# Set up global R environment

#   options(warn=-1)
#   sink("myfile", append=FALSE, split=FALSE)
#   options(repr.plot.width=12, repr.plot.height=8, jupyter.plot_mimetypes = 'image/png')

# Install neccesary R packages used by the functions in the library

   list.of.packages <- c("scatterplot3d", "MASS", "RColorBrewer", "smacof", "NMF",
                         "maptools", "tensor", "fastcluster", "spatstat", "e1071", "Rtsne", "rgr",
                         "verification", "sn", "misc3d", "PerformanceAnalytics", "NMF", "ppcor",
                         "gtools", "fMultivar", "gmodels")
                       
   packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
   if(length(packages)) install.packages(packages)

   suppressMessages(library(scatterplot3d))
   suppressMessages(library(MASS))
   suppressMessages(library(RColorBrewer))
#   suppressMessages(library(rgl))
#   suppressMessages(library(bpca))
   suppressMessages(library(smacof))
   suppressMessages(library(NMF))
   suppressMessages(library(maptools))
   suppressMessages(library(tensor))
   suppressMessages(library(fastcluster))
   suppressMessages(library(spatstat))
   suppressMessages(library(e1071))
   suppressMessages(library(Rtsne))
   suppressMessages(library(rgr))
   suppressMessages(library(verification))
   suppressMessages(library(sn))
   suppressMessages(library(misc3d))
   suppressMessages(library(PerformanceAnalytics))
   suppressMessages(library(NMF))
   suppressMessages(library(ppcor))
   suppressMessages(library(gtools))
   suppressMessages(library(fMultivar))
   suppressMessages(library(gmodels))

#-------------------------------------------------------------------------------------------------	
# Function definitions
#
#-------------------------------------------------------------------------------------------------
   CCBA_compute_assoc_or_dist.v1 <- function( 
   #
   #  Computes the distance between columns or rows across two matrices (two input datasets)
   #  Pablo Tamayo Dec 30, 2015
   #                                            
      input_matrix1,                # Input matrix 1
      input_matrix2,                # Input matrix 2
      object_type     = "columns",  # Object type "columns" or "rows"
      assoc_metric    = "IC",       # Association metric: "IC", "ICR", "COR", "SPEAR"
      distance        = T)          # Compute distance (T) or association (F)
   {
      if (object_type == "rows") {
         input_matrix1 <- t(input_matrix1)
         input_matrix2 <- t(input_matrix2)
      }
      assoc.matrix <- matrix(0, nrow=ncol(input_matrix1), ncol=ncol(input_matrix2),
                             dimnames = list(colnames(input_matrix1), colnames(input_matrix2)))
      for (i in 1:ncol(input_matrix1)) {
         for (j in 1:ncol(input_matrix2)) {
            if (assoc_metric == "IC") {
               assoc.matrix[i, j] <- CCBA_IC.v1(input_matrix1[,i], input_matrix2[,j])
            } else if (assoc_metric == "ICR") {
               assoc.matrix[i, j] <- CCBA_IC.v1(rank(input_matrix1[,i]), rank(input_matrix2[,j]))
            } else if (assoc_metric == "COR") {
               assoc.matrix[i, j] <- cor(input_matrix1[,i], input_matrix2[,j], method = "pearson")
            } else if (assoc_metric == "SPEAR") {
               assoc.matrix[i, j] <- cor(input_matrix1[,i], input_matrix2[,j], method = "spearman")
            } else {
               stop(paste("ERROR: unknown association metric:", assoc_metric))
           }
          }
      }
      if (distance == T) {
         return(as.dist(1 - assoc.matrix))
      } else {
         return(assoc.matrix)
      }
   }

#-------------------------------------------------------------------------------------------------
   CCBA_read_GCT_file.v1 <- function(filename = "NULL") 
   #    
   # Reads a gene expression dataset in GCT format and converts it into an R data frame
   # Pablo Tamayo Dec 30, 2015
   #
   {
      ds <- read.delim(filename, header=T, sep="\t", skip=2, row.names=1, blank.lines.skip=T,
                       comment.char="", as.is=T, na.strings = "")
      descs <- ds[,1]
      ds <- ds[-1]
      row.names <- row.names(ds)
      names <- names(ds)
      return(list(ds = ds, row.names = row.names, descs = descs, names = names))
   }

#------------------------------------------------------------------------------------------------
   CCBA_ReadClsFile <- function(file = "NULL") 
   #
   # Reads a class vector CLS file and defines phenotype and class labels vectors (numeric and character)
   # for the samples in a gene expression file (RES or GCT format)
   #
    {
      cls.cont <- readLines(file)
      num.lines <- length(cls.cont)
      class.list <- unlist(strsplit(cls.cont[[3]], " "))
      s <- length(class.list)
      t <- table(class.list)
      l <- length(t)
      phen <- vector(length=l, mode="character")
      class.v <- vector(length=s, mode="numeric")
     
      current.label <- class.list[1]
      current.number <- 1
      class.v[1] <- current.number
      phen[1] <- current.label
      phen.count <- 1

      if (length(class.list) > 1) {
         for (i in 2:s) {
             if (class.list[i] == current.label) {
                  class.v[i] <- current.number
             } else {
                  phen.count <- phen.count + 1
                  current.number <- current.number + 1
                  current.label <- class.list[i]
                  phen[phen.count] <- current.label
                  class.v[i] <- current.number
             }
        }
       }
     return(list(phen = phen, class.v = class.v, class.list = class.list))
}

#-------------------------------------------------------------------------------------------------
   CCBA_make_heatmap.v1 <- function(
   #    
   # Make heatmap of matrix from GCT file
   # Pablo Tamayo Dec 30, 2015
   #                                
      input_dataset,                 # Input dataset (GCT). This is e.g. the original dataset A or the H matrix
      annot_file = NULL,             # Phenotype annotation file (TXT, optional) in format c(file, name_column, annot_column, use_prefix, color_column or NULL)
      transpose_data = F,            # Transpose input matrix
      append_annot = F,              # Append annotation to column names
      sort_columns = T,              # Sort columns in heatmap
      sort_rows = T,                 # Sort rows in heatmap
      order_phenotypes_by_size = F,  # Order phenotypes by size
      produce_average_over_phen_heatmap = F, # Produce a heatmap of average values per phenotype
      cex.rows = "auto",             # Character size for row names
      cex.cols = "auto",             # Character size for col names
      cex.phen = "auto",             # Character size for phenotype names
      str.phen = "90",               # Degrees: orientation of phenotype labels              
      left.margin = 15,              # Left margin for plot
      debug_mode = F,                # Debug mode
      output.pdf = NULL)             # Dataset to save output plot
   {

      set.seed(5209761)

      if (!is.null(output.pdf)) pdf(file=output.pdf, height=8.5, width=11)
      
      # Read input dataset

      dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
      H <- data.matrix(dataset.1$ds)
      if (debug_mode == T) print(dim(H))
      if (debug_mode == T) print(H)      

      if (debug_mode == T)    print("Components (row) sum amplitudes:")
      if (debug_mode == T)    print(cbind(sort(apply(H, MARGIN=1, FUN=sum), decreasing=T)))

      if (transpose_data == T) H <- t(H)

      # Defaults colors

      phen.col <-   c("plum3", "steelblue2", "seagreen3", "orange", "indianred3",  "cyan3", brewer.pal(7, "Set1"), brewer.pal(7,"Dark2"), 
                            brewer.pal(7, "Set1"),  brewer.pal(7, "Paired"), brewer.pal(8, "Accent"), brewer.pal(8, "Set2"),
                            brewer.pal(11, "Spectral"), brewer.pal(12, "Set3"))
         
      # Read annotation file
      
      if (!is.null(annot_file)) {
         annot.table <- read.table(annot_file[[1]], header=T, sep="\t", skip=0, colClasses = "character")
         column.list <- annot.table[, annot_file[[2]]]
         annot.list <- annot.table[, annot_file[[3]]]
         column.set <- vector(length=ncol(H), mode="character")
         if (annot_file[[4]] == T) {
            for (i in 1:ncol(H)) {
               column.set[i] <- strsplit(colnames(H)[i], split="_")[[1]]
            }
         } else {
            column.set <- colnames(H)
         }
         locs <- match(column.set, column.list)
      if (debug_mode == T) print(length(column.set))
      if (debug_mode == T)  print(length(column.list))      
      if (debug_mode == T)  print(paste(length(locs[!is.na(locs)]), " samples match the annotation table"))
         column.class <- annot.list[locs]
         column.class[is.na(column.class)] <- "UNLABELED"
         all.classes <- unique(column.class)
      if (debug_mode == T)  print(table(column.class))
         if (append_annot == T) colnames(H) <- paste(colnames(H), " (", column.class, ") ", sep="")

      # is there a color for each annotation
      if (length(annot_file) > 4) {
         if (!is.null(annot.table[, annot_file[[5]]])) {
           phen.col <- annot.table[, annot_file[[5]]]
         }
      }
      
   } else {
      column.class <- rep(" ", ncol(H))
      all.classes <- " "
   }
  
   # Color map
 
   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   max.cont.color <- 511
   mycol <- c(mycol,
              "black",  # brewer.pal(9, "YlGn")[1],     # Missing feature color (light yellow)
              mycol[256 - 75],                          # Binary feature's 0's color (blue)
              mycol[256 + 220])                         # Binary feature's 1's color (red)
   cex.axis = 1

    # Normalize and apply color map

   cutoff <- 2
   for (i in 1:nrow(H)) {
      x <- H[i,]
      locs.non.na <- !is.na(x)
      x.nonzero <- x[locs.non.na]
      x.nonzero2 <- (x.nonzero - mean(x.nonzero))/sd(x.nonzero)         
      x.nonzero2[x.nonzero2 > cutoff] <- cutoff
      x.nonzero2[x.nonzero2 < - cutoff] <- - cutoff      
      s <- strsplit(row.names(H)[i], "_")[[1]]
      suffix <- s[length(s)]
      if (suffix == "MUT" | suffix == "AMP" | suffix == "DEL" | suffix == "AMP_2" | suffix == "AMP_3" | suffix == "DEL_2" | suffix == "DEL_3" |
          suffix == "all" | length(table(x.nonzero)) == 2) {  # Binary feature
         H[i,locs.non.na] <- x.nonzero + max.cont.color + 2   # binary feature colors
       } else {
         H[i, locs.non.na] <- x.nonzero2
         H[i, locs.non.na] <- ceiling(max.cont.color * (H[i,locs.non.na] + cutoff)/(2*cutoff))
         H[i, locs.non.na] <- ifelse (H[i, locs.non.na] > max.cont.color, max.cont.color, H[i, locs.non.na])
       }
      H[i, is.na(x)] <- max.cont.color + 1 # missing feature color 
    }

   # Plot sorted by phenotype 

   nf <- layout(matrix(c(1, 2, 3), nrow=3, ncol=1, byrow=T), 1, c(1, 6.5, 0.75), FALSE)

   H_orig <- H
   column.class_orig <- column.class

   if (produce_average_over_phen_heatmap == F) {
   
   if (order_phenotypes_by_size == T) {

      num.class <- match(column.class, all.classes)
      sizes <- table(num.class)
      ind.sizes <- order(sizes, decreasing=T)
      seq.ind <- seq(1, length(sizes))
      num.class <- match(num.class, ind.sizes)
      ind <- order(num.class, decreasing=F)
      H <- H[, ind]
      column.class <- column.class[ind]
      all.classes <- unique(column.class)

   } else {
      ind <- order(column.class, decreasing=F)
      H <- H[, ind]
      column.class <- column.class[ind]
      all.classes <- sort(unique(column.class))
   }
   
   V1.phen <- match(column.class, all.classes)
   par(mar = c(1, left.margin, 2, 6))
   image(1:length(V1.phen), 1:1, as.matrix(V1.phen), col=phen.col[1:max(V1.phen)], axes=FALSE, main="Phenotype", sub = "", xlab= "", ylab="")
   axis(2, at=1:1, labels="Phenotype", adj= 0.5, tick=FALSE, las = 1, cex.axis=1, font.axis=1, line=-1)

   if (!is.null(annot_file)) {
      leg.txt <- all.classes
      for (i in 1:length(leg.txt)) leg.txt[i] <- substr(leg.txt[i], 1, 25)
      boundaries <- NULL
      for (i in 2:length(column.class)) {
         if (column.class[i] != column.class[i-1]) boundaries <- c(boundaries, i-1)
      }
      boundaries <- c(boundaries, length(column.class))
      locs.bound <- c(boundaries[1]/2, boundaries[2:length(boundaries)]
                   - (boundaries[2:length(boundaries)] - boundaries[1:(length(boundaries)-1)])/2)
      sizes <- boundaries - c(1, boundaries[1:(length(boundaries) -1)])
   if (debug_mode == T) print(sizes)
      for (i in 1:length(leg.txt)) {
         if (cex.phen == "auto") {
             cex <- 0.05 + 7/ifelse(nchar(leg.txt[i]) < 6, 6, nchar(leg.txt[i]))
         } else if (cex.phen == "auto2") {
             cex <- 0.40 + 0.125*sizes[i]/nchar(leg.txt[i])
   if (debug_mode == T) print(cex)
         } else {
            cex <- cex.phen
        }
        text(locs.bound[i] + 0.5, 1, labels=leg.txt[i], adj=c(0.25, 0.5), srt=str.phen, cex=cex)
       }
    }
   
  # Sort columns inside each phenotypic class 
   
   for (k in all.classes) {
      if (sum(column.class == k) <= 1) next;
      V1 <- H[, column.class == k]
      col.names.V1 <- colnames(H)[column.class == k]
      dist.matrix <- CCBA_compute_assoc_or_dist.v1(input_matrix1 = V1, input_matrix2 = V1, object_type = "columns",
                                                     assoc_metric = "IC", distance = T)
      HC <- hclust(dist.matrix, method="ward.D2")
      ind <- HC$order
      V1 <- V1[ , ind]
      H[, column.class == k] <- V1
      col.names.V1 <- col.names.V1[ind]
      colnames(H)[column.class == k] <- col.names.V1 
   }
 
   V2 <- apply(H, MARGIN=2, FUN=rev)
   lower.space <-  ceiling(4 + 100/nrow(H))
   par(mar = c(lower.space, left.margin, 2, 6))
   image(1:ncol(V2), 1:nrow(V2), t(V2), zlim = c(0, max.cont.color + 3), col=mycol, axes=FALSE,
         main="Matrix Sorted by Phenotype",  sub = "", xlab= "", ylab="", cex.main=1)
   if (cex.rows == "auto") cex.rows <- 0.20 + 200/(nrow(V2) * max(nchar(row.names(V2))) + 200)

   axis(2, at=1:nrow(V2), labels=row.names(V2), adj= 0.5, tick=FALSE, las = 1, cex.axis=cex.rows, font.axis=1, line=-1)
   if (!is.null(annot_file)) {   
      for (i in 1:(length(boundaries)-1)) lines(c(boundaries[i]+0.5, boundaries[i]+0.5),
                                                c(0.5, nrow(V2) + 0.5), lwd=2, lty=1, col="black")
    }
   if (!is.null(annot_file)) {
      cols2 <- phen.col[match(column.class, all.classes)]
   } else {
      cols2 = "black"
   }
   if (cex.cols == "auto") cex.cols <- 0.20 + 200/(ncol(V2) * max(nchar(colnames(V2))) + 200)
   mtext(colnames(V2), at=1:ncol(V2), side = 1, cex=cex.cols, col=cols2, line=0, las=3, font=2, family="")
   
   # Legend

   par(mar = c(3, 35, 1, 6))
   leg.set <- seq(-cutoff, cutoff, 0.05)
   image(1:length(leg.set), 1:1, as.matrix(leg.set), zlim=c(-cutoff, cutoff), col=mycol, axes=FALSE, main="Matrix Standardized Profile",
       sub = "", xlab= "", ylab="",font=2, family="", mgp = c(0, 0, 0), cex.main=0.8)
   ticks <- seq(-cutoff, cutoff, 0.5)
   tick.cols <- rep("black", 5)
   tick.lwd <- 1
   locs <- NULL
   for (k in 1:length(ticks)) locs <- c(locs, which.min(abs(ticks[k] - leg.set)))
   axis(1, at=locs, labels=ticks, adj= 0.5, tick=T, cex=0.6, cex.axis=0.6, line=0, font=2, family="", mgp = c(0.1, 0.1, 0.1))


  } else if (produce_average_over_phen_heatmap == T) {

     H.mean <- NULL
     for (k in 1:nrow(H)) {
        H.row <- unlist(lapply(split(H[k,], V1.phen), mean))
        H.mean <- rbind(H.mean, H.row)
     }
     row.names(H.mean) <- row.names(H)
     colnames(H.mean) <- all.classes

     nf <- layout(matrix(c(1, 2, 3), nrow=3, ncol=1, byrow=T), 1, c(1, 6.5, 0.75), FALSE)
     par(mar = c(1, left.margin, 2, 6))

     image(1:ncol(H.mean), 1:1, as.matrix(1:ncol(H.mean)), col=phen.col[1:ncol(H.mean)], axes=FALSE, main="Phenotype", sub = "", xlab= "", ylab="")
     axis(2, at=1:1, labels="Phenotype", adj= 0.5, tick=FALSE, las = 1, cex.axis=1, font.axis=1, line=-1)

     for (i in 1:ncol(H.mean))  text(i, 1, labels=all.classes[i], adj=c(0.25, 0.5), srt=str.phen, cex=cex)
     
     VX <- apply(H.mean, MARGIN=2, FUN=rev)
     lower.space <-  ceiling(4 + 100/nrow(H.mean))
     par(mar = c(lower.space, 10, 2, 6))

     image(1:ncol(VX), 1:nrow(VX), t(VX), zlim = c(0, max.cont.color + 3), col=mycol, axes=FALSE, main="Matrix Averaged by Phenotype",
         sub = "", xlab= "", ylab="", cex.main=1)

     axis(2, at=1:nrow(VX), labels=row.names(VX), adj= 0.5, tick=FALSE, las = 1, cex.axis=cex.rows, font.axis=1, line=-1)

   # Legend

     par(mar = c(3, 35, 1, 6))
     leg.set <- seq(-cutoff, cutoff, 0.05)
     image(1:length(leg.set), 1:1, as.matrix(leg.set), zlim=c(-cutoff, cutoff), col=mycol, axes=FALSE, main="Matrix Standardized Profile",
       sub = "", xlab= "", ylab="",font=2, family="", mgp = c(0, 0, 0), cex.main=0.8)
     ticks <- seq(-cutoff, cutoff, 0.5)
     tick.cols <- rep("black", 5)
     tick.lwd <- 1
     locs <- NULL
     for (k in 1:length(ticks)) locs <- c(locs, which.min(abs(ticks[k] - leg.set)))
     axis(1, at=locs, labels=ticks, adj= 0.5, tick=T, cex=0.6, cex.axis=0.6, line=0, font=2, family="", mgp = c(0.1, 0.1, 0.1))
    }

   if (!is.null(output.pdf)) dev.off()
      
  }

#-------------------------------------------------------------------------------------------------
   CCBA_IC.v1 <-  function(x, y, n.grid=25)  
   #
   # Compute Information Coefficient [IC]
   # Pablo Tamayo Dec 30, 2015
   #    
  {
      x.set <- !is.na(x)
      y.set <- !is.na(y)
      overlap <- x.set & y.set

      x <- x[overlap] +  0.000000001*runif(length(overlap))
      y <- y[overlap] +  0.000000001*runif(length(overlap))

      if (length(x) > 2) {
         delta = c(bcv(x), bcv(y))
         rho <- cor(x, y)
         rho2 <- abs(rho)
         delta <- delta*(1 + (-0.75)*rho2)
         kde2d.xy <- kde2d(x, y, n = n.grid, h = delta)
         FXY <- kde2d.xy$z + .Machine$double.eps
         dx <- kde2d.xy$x[2] - kde2d.xy$x[1]
         dy <- kde2d.xy$y[2] - kde2d.xy$y[1]
         PXY <- FXY/(sum(FXY)*dx*dy)
         PX <- rowSums(PXY)*dy
         PY <- colSums(PXY)*dx
         HXY <- -sum(PXY * log(PXY))*dx*dy
         HX <- -sum(PX * log(PX))*dx
         HY <- -sum(PY * log(PY))*dy
         PX <- matrix(PX, nrow=n.grid, ncol=n.grid)
         PY <- matrix(PY, byrow = TRUE, nrow=n.grid, ncol=n.grid)
         MI <- sum(PXY * log(PXY/(PX*PY)))*dx*dy
         IC <- sign(rho) * sqrt(1 - exp(- 2 * MI))
         if (is.na(IC)) IC <- 0
      } else {
         IC <- 0
      }
      return(IC)
   }

#-------------------------------------------------------------------------------------------------
   CCBA_OncoGPS_define_states.v1 <- function(
   # 
   # Computes the OncoGPS states by consensus clustering
   # P. Tamayo Jan 17, 2016
   #    
      input_dataset,                               # Input GCT dataset (e.g. an H matrix from NMF decomposition)
      norm_thres                       =  3,       # Normalization threhold
      output_dist_matrix_file          = NULL,     # Filename to save distance matrix
      k_min                            = 2,        # Min number of states
      k_max                            = 5,        # Max number of states
      k_incr                           = 1,        # Increment for number of states
      num_clusterings                  = 3,        # Number on instances for consensus clustering
      random_seed                      = 5209761,  # RNG seed
      membership_matrix_file,                      # Membership matrix output file (GCT)
      output_plots)                                # Output PDF file with NMF plots
 {


    set.seed(random_seed)
   
   mycol <- vector(length=512, mode = "numeric")   # Red/Blue "pinkogram" color map
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)
    
   # Read expression dataset

   dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   H <- data.matrix(dataset.1$ds)
   print(paste("Dimensions matrix A:", nrow(H), ncol(H)))

   print(dim(H))
   flush.console()
   for (k in 1:nrow(H)) {
      row.mean <- mean(H[k,])
      row.sd <- sd(H[k,])
      H[k,] <- (H[k,] - row.mean)/row.sd
      for (j in 1:ncol(H)) {
         if (H[k, j] > norm_thres)  H[k, j] <- norm_thres
            if (H[k, j] < -norm_thres) H[k, j] <- -norm_thres
      }
   }
   nodes.min <- apply(H, MARGIN=1, FUN=min)
   nodes.max <- apply(H, MARGIN=1, FUN=max)       
   for (j in 1:nrow(H)) H[j,] <- (H[j,] - nodes.min[j])/(nodes.max[j] - nodes.min[j])

   # Compute distance matrix using the Information Coefficient as the metric

   print("H:")
   print(H[, 1:25])
   
   dist.matrix <- CCBA_compute_assoc_or_dist.v1(input_matrix1 = t(H), input_matrix2 = t(H), object_type = "rows",
                                                assoc_metric = "IC", distance = T)
   dist.matrix.out <- as.matrix(dist.matrix)
   if (!is.null(output_dist_matrix_file)) {
      print("Saving distance matrix")          
      CCBA_write.gct.v1(gct.data.frame = dist.matrix.out, descs = row.names(dist.matrix), filename = output_dist_matrix_file)
   }
   
   # Perform consensus clustering

   num_k <- length(seq(k_min, k_max, k_incr))
   rho <- k_vector <- vector(mode = "numeric", length = num_k)
   connect.matrix.ordered <- array(0, c(num_k, ncol(H), ncol(H)))
   all.membership <- matrix(NA, nrow=ncol(H), ncol=num_k)
   row.names(all.membership) <- colnames(H)
   colnames(all.membership) <- paste("k_", seq(k_min, k_max, k_incr), sep="")
   
   k_index <- 1
   
   for (k in seq(k_min, k_max, k_incr)) { 
      assign <- matrix(0, nrow = num_clusterings, ncol = ncol(H))
      for (i in 1:num_clusterings) {

         # Bootstrap sampling

         locs <- sample(seq(1, ncol(H)), ncol(H), replace = T)
         dist.matrix.ins <- dist.matrix.out[locs, locs]
         sample.names <- colnames(dist.matrix.out)[locs]
         dist.matrix.ins <- as.dist(dist.matrix.ins)
         HC <- hclust(dist.matrix.ins, method="ward.D")
         cutree.model <- cutree(HC, k = k, h = NULL)
         locs2 <- match(sample.names, colnames(dist.matrix.out))
         assign[i, locs] <- cutree.model
         cutree.model <- paste("S", cutree.model, sep="")
         # print(paste("Table of states membership: (from k=", k, " classes clustering)"))
         # print(table(cutree.model))
        }
   
     # Compute consensus matrix
      
     connect.matrix <- matrix(0, nrow = ncol(H), ncol = ncol(H))

     for (i in 1:num_clusterings) {
       for (j in 1:ncol(H)) {
          for (p in 1:ncol(H)) {
             if (j != p) {
                 if ((assign[i, j] != 0) & (assign[i, p] != 0)) {
                    if (assign[i, j] == assign[i, p]) {
                       connect.matrix[j, p] <- connect.matrix[j, p] + 1
                    }
                  }
              } else {
                    connect.matrix[j, p] <- connect.matrix[j, p] + 1
              }
           }
       }
     }
     connect.matrix <- connect.matrix/num_clusterings
     cons.dist.matrix <- 1 - connect.matrix
     cons.dist.matrix <- as.dist(cons.dist.matrix)
     HC <- hclust(cons.dist.matrix, method="ward.D")
     dist.coph <- cophenetic(HC)
     k_vector[k_index] <- k
     rho[k_index] <- cor(cons.dist.matrix, dist.coph)
     rho[k_index] <- signif(rho[k_index], digits = 4)
     for (i in 1:ncol(H)) {
        for (j in 1:ncol(H)) {
           connect.matrix.ordered[k_index, i, j] <- connect.matrix[HC$order[i], HC$order[j]]
         }
     }

     # Compute consensus clustering membership

     membership <- paste("S", cutree(HC, k = k), sep="")
     all.membership[, k_index] <- membership
     k_index <- k_index + 1

   } # loop over k

#   CCBA_write.gct.v1(gct.data.frame = all.membership, descs = row.names(all.membership), membership_matrix_file)

   col.names <- paste(colnames(all.membership), collapse = "\t")
   col.names <- paste("SAMPLE", col.names, sep= "\t")
   write(noquote(col.names), file = membership_matrix_file, append = F, ncolumns = length(col.names))
   write.table(all.membership, file=membership_matrix_file, quote=F, col.names = F, row.names = T, append = T, sep="\t")
   
   print("Number of samples in each state:")
   for (k in 1:ncol(all.membership)) {
      print("-------------------------------------")
      print(paste("k=", k_vector[k]))
      print(table(all.membership[, k]))
   }
   
   peak <- rep(0, length(k_vector))

   pdf(file=output_plots, height=8.5, width=11)

   plot(k_vector, rho, type="n")
   points(k_vector, rho, type="l")
   points(k_vector, rho, type="p", pch=20)          

#   for (h in 2:(length(rho) - 1)) if (rho[h - 1] < rho[h] & rho[h] > rho[h + 1]) peak[h] <- 1
#   k_peaks <- k_vector[peak == 1]
#   k <- rev(k_peaks)[1]

   dev.off()

 }

#-------------------------------------------------------------------------------------------------
   CCBA_OncoGPS_create_map.v1 <- function(
   # 
   # Generates the 2D OncoGPS map
   # P. Tamayo Jan 17, 2016
   #    
       reference_dataset, 
       states_table,
       sample_names_column,
       states_column,
       node.nicknames          = NULL,
       size.grid               = 150, 
       ncol                    = 24, 
       color.factor            = 1.0, 
       plot_sample_names       = F,
       description             = NULL,
       point.cex               = 2,
       kernel.width            = 0.20,
       contour.tone            = 5,
       contour.levels          = 10,
       low.range.color         = 2,
       high.range.color        = 6,
       norm.thres              = 3,
       expon                   = 2,
       row.projection.method   = "sam",               
       output.file             = NULL,
       states.file             = NULL,
       state.col.pal           = 2,   # state color palette # 1 or 2 
       OPM.objects.file)
    {

   set.seed(5209761)

   # Color map
 
   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   max.cont.color <- 511
   mycol <- c(mycol,
              "black",  # brewer.pal(9, "YlGn")[1],     # Missing feature color (light yellow)
              mycol[256 - 75],                          # Binary feature's 0's color (blue)
              mycol[256 + 220])                         # Binary feature's 1's color (red)
   cex.axis = 1
   binary.col <- c("lightgray", # zero color
                   "black")     # one color
   mycol.class <- matrix(0, nrow=ncol, ncol=ncol)

   if (state.col.pal == 1) {
   
        categ.col2 <- c(
                      brewer.pal(9, "Blues")[6], brewer.pal(9, "Greens")[6], brewer.pal(9, "Reds")[6], brewer.pal(9, "Purples")[6],          
                     "#9DDDD6", # dusty green
                     "#F0A5AB", # dusty red
                     "#9AC7EF", # sky blue
                     "#D6A3FC", # purple
                     "#FFE1DC", # clay
                     "#FAF2BE", # dusty yellow
                     "#F3C7F2", # pink
                     "#C6FA60", # green
                     "#F970F9", # violet
                     "#FC8962", # red                     
                     "#F6E370", # orange
                     "#F0F442", # yellow
                     "#AED4ED", # steel blue
                     "#D9D9D9", # grey
                     "#FD9B85", # coral
                     "#7FFF00", # chartreuse
                     "#FFB90F", # goldenrod1
                     "#6E8B3D", # darkolivegreen4
                     "#8B8878", # cornsilk4
                     "#7FFFD4", # aquamarine
                     "darkblue",
                     "tan",
                     "darkgreen")

      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Blues")[6])) # blue
      mycol.class[,1] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Greens")[6])) # green   
      mycol.class[,2] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Reds")[6])) # red   
      mycol.class[,3] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Purples")[6])) # purple   
      mycol.class[,4] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#2D6D66")) # dusty green   
      mycol.class[,5] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#90353B")) # dusty red
      mycol.class[,6] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#2A577F")) # sky blue
      mycol.class[,7] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#6633CC")) # purple
      mycol.class[,8] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#8F716C")) # clay 
      mycol.class[,9] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#9A924E")) # dusty yellow
      mycol.class[,10] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#E377C2")) # pink
      mycol.class[,11] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#569A00")) # green
      mycol.class[,12] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#990099")) # violet
      mycol.class[,13] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#CC2902")) # red
      mycol.class[,14] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#E67300")) # orange
      mycol.class[,15] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#F0E442")) # yellow
      mycol.class[,16] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#9EC4DD")) # steel blue
      mycol.class[,17] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#A9A9A9")) # grey
      mycol.class[,18] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#CD5B45")) # coral
      mycol.class[,19] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#7FFF00")) # chartreuse
      mycol.class[,20] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#FFB90F")) # goldenrod1
      mycol.class[,21] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "darkblue")) 
      mycol.class[,22] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "tan")) 
      mycol.class[,23] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "darkgreen")) 
      mycol.class[,24] <- colfunc(color.factor*ncol)[1:ncol]

    } else {
      categ.col2 <- c(
                     "plum3",
                     "steelblue2",
                     "seagreen3",
                     "orange",
                     "indianred3",
                     "#F0A5AB", # dusty red
                     "#9AC7EF", # sky blue
                     "#D6A3FC", # purple
                     "#FFE1DC", # clay
                     "#FAF2BE", # dusty yellow
                     "#F3C7F2", # pink
                     "#C6FA60", # green
                     "#F970F9", # violet
                     "#FC8962", # red                     
                     "#F6E370", # orange
                     "#F0F442", # yellow
                     "#AED4ED", # steel blue
                     "#D9D9D9", # grey
                     "#FD9B85", # coral
                     "#7FFF00", # chartreuse
                     "#FFB90F", # goldenrod1
                     "#6E8B3D", # darkolivegreen4
                     "#8B8878", # cornsilk4
                     "#7FFFD4", # aquamarine
                     "darkblue",
                     "tan",
                     "darkgreen")

      colfunc <- colorRampPalette(c("white", "plum3")) 
      mycol.class[,1] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "steelblue2")) 
      mycol.class[,2] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "seagreen3")) 
      mycol.class[,3] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "orange")) 
      mycol.class[,4] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "indianred3")) 
      mycol.class[,5] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#90353B")) # dusty red
      mycol.class[,6] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#2A577F")) # sky blue
      mycol.class[,7] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#6633CC")) # purple
      mycol.class[,8] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#8F716C")) # clay 
      mycol.class[,9] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#9A924E")) # dusty yellow
      mycol.class[,10] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#E377C2")) # pink
      mycol.class[,11] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#569A00")) # green
      mycol.class[,12] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#990099")) # violet
      mycol.class[,13] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#CC2902")) # red
      mycol.class[,14] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#E67300")) # orange
      mycol.class[,15] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#F0E442")) # yellow
      mycol.class[,16] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#9EC4DD")) # steel blue
      mycol.class[,17] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#A9A9A9")) # grey
      mycol.class[,18] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#CD5B45")) # coral
      mycol.class[,19] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#7FFF00")) # chartreuse
      mycol.class[,20] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "#FFB90F")) # goldenrod1
      mycol.class[,21] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "darkblue")) 
      mycol.class[,22] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "tan")) 
      mycol.class[,23] <- colfunc(color.factor*ncol)[1:ncol]
      colfunc <- colorRampPalette(c("white", "darkgreen")) 
      mycol.class[,24] <- colfunc(color.factor*ncol)[1:ncol]

   }
   myncolors <- ncol

   if (!is.null(output.file))  pdf(file=output.file, height=8.5, width=11)

   dataset <- CCBA_read_GCT_file.v1(filename = reference_dataset)
   m.1 <- data.matrix(dataset$ds)
   sample.names.1 <- colnames(m.1)
   Ns.1 <- ncol(m.1)
   print(paste("Total samples in input file:", Ns.1))

   if (point.cex == "auto") point.cex <- ifelse(Ns.1 < 10, 1.5, (1.5 - 10/990) + 1/990*Ns.1)

   nodes <- row.names(m.1)
   if (is.null(node.nicknames)) node.nicknames <- nodes

   H <- m.1[nodes,]

   print("Performing normalization")

   nodes.mean <- apply(H, MARGIN=1, FUN=mean)
   nodes.sd   <- apply(H, MARGIN=1, FUN=sd)
   for (i in 1:nrow(H)) {
       H[i,] <- (H[i,] - nodes.mean[i])/nodes.sd[i]
       for (j in 1:ncol(H)) {
          if (H[i, j] >  norm.thres) H[i, j] <- norm.thres
          if (H[i, j] < -norm.thres) H[i, j] <- -norm.thres
        }
   }
   nodes.min <- apply(H, MARGIN=1, FUN=min)
   nodes.max <- apply(H, MARGIN=1, FUN=max)
   for (i in 1:nrow(H)) {   
       H[i,] <- (H[i,] - nodes.min[i])/(nodes.max[i] - nodes.min[i])           
   }

   print(dim(H))   
   print(H[,1:10])
   flush.console()
   
   # Read states from states file

    print("Reading pre-defined states")      

   class.table <- read.delim(states_table, header=T, row.names=NULL, sep="\t", skip=0)
   class.table.samples <- class.table[, sample_names_column]
   class.table.classes <- as.character(class.table[, states_column])

   print("states_column:")
   print(states_column)
   
   print("class.table.samples:")
   print(class.table.samples[1:10])

   print("class.table.classes:")
   print(class.table.classes[1:10])
   
   print("Matching sample names")
   print(colnames(H)[1:3])

   flush.console()   

   locs <- match(colnames(H), class.table.samples)
   if (sum(is.na(locs) > 0)) {
       print("ERROR: samples mismatch against the ones in states file")
       print(locs)
       print(colnames(H)[is.na(locs)])
       stop("STOP:ERROR: samples mismatch against the ones in states ile ")
   }
   cutree.model <- class.table.classes[locs]
   all.classes <- unique(cutree.model)
   k.classes <- length(all.classes)

   print(cutree.model[1:10])

   print("Table of states membership: (from states file)")
   print(table(cutree.model))

   # Make logistic predictive model for each stateg

   print("Making logistic predictive model for each state")
   flush.console()
   
   svm.mod <- svm(x = t(H), y = cutree.model, type = "C-classification")
   predicted.state <- predict(svm.mod, newdata = t(H))

   print("Training confussion matrix")
   print(table(predicted.state, cutree.model))
      
   col <- categ.col2[match(cutree.model, all.classes)]
   
   states.names <- all.classes

   # Save states in a GCT file for annotation purposes

   states <- matrix(0, nrow=k.classes, ncol=ncol(H), dimnames=list(states.names, colnames(H)))
   for (k in all.classes) {
      states[k, cutree.model == k] <- 1
   }
   all.features <- rbind(cutree.model, match(cutree.model, all.classes), states, as.character(predicted.state))
   row.names(all.features) <- c("states", "numerical.states", states.names, "predicted_state")

   if (!is.null(states.file)) {
      CCBA_write.gct.v1(gct.data.frame = as.matrix(all.features), descs = row.names(all.features), filename = states.file)
   }

   # Project rows 

   print("Projecting rows")      

   if (length(nodes) > 3) {

      proj <- gx.2dproj(H, proc = row.projection.method, ifilr = F, log = FALSE, rsnd = F, snd = F,   # proc= sam, ica or iso       
                        range = FALSE, main = "", setseed = 123456, row.omits = NULL)
      row.objects <- cbind(proj$x, proj$y)
      row.objects <- (row.objects - min(row.objects))/(max(row.objects) - min(row.objects))
      print(row.objects[1:3,])
      flush.console()   

      # Define simplexes using delaunay triangulation
     
      X <- ppp(row.objects[,1], row.objects[,2], range(row.objects[,1]), range(row.objects[,2]))
      X2 <- delaunay(X)
      X3 <- tiles(X2)
      triangles.matrix <- matrix(NA, nrow=length(X3), ncol=3)
      for (tri in 1:length(X3)) {
         X4 <- unlist(X3[[tri]])
         node1 <- c(X4["bdry.x1"], X4["bdry.y1"])
         node2 <- c(X4["bdry.x2"], X4["bdry.y2"])
         node3 <- c(X4["bdry.x3"], X4["bdry.y3"])            

         for (i in 1:nrow(row.objects)) { 
           if (sum(node1 == row.objects[i,]) == 2) node.iden1 <- i
           if (sum(node2 == row.objects[i,]) == 2) node.iden2 <- i
           if (sum(node3 == row.objects[i,]) == 2) node.iden3 <- i
        }
        triangles.matrix[tri,] <- c(node.iden1, node.iden2, node.iden3)
     }
     triangles <- NULL
     for (tri in 1:length(X3)) {
        triangles <- c(triangles, list(triangles.matrix[tri,]))
     }
     w <- convexhull.xy(X)

  } else { # one triangle
    triangles <- list(c(1, 2, 3))
    row.objects <- matrix(c(  0,    0,       # node 1
                            0.5, sqrt(3)/2,  # node 2
                              1,    0),      # node 3
                            nrow=3, ncol=2, byrow=T)
    X <- ppp(row.objects[,1], row.objects[,2], range(row.objects[,1]), range(row.objects[,2]))
    w <- convexhull.xy(X)    
   }

   xp <- yp <- xp.local <- yp.local <- rep(0, ncol(H))   

   H.prime <- H
   H.prime2 <- matrix(0, nrow=nrow(H.prime), ncol=ncol(H.prime))
   for (j in 1:ncol(H.prime2)) {
      top.vals <- order(H.prime[,j], decreasing=T)[1:3]
      H.prime2[top.vals,j] <- H.prime[top.vals,j]
   }

   # project columns objects
   
   for (j in 1:ncol(H)) {
       weight <- sum(H.prime2[,j]^expon)
       xp[j] <- sum(H.prime2[,j]^expon*row.objects[,1])/weight
       yp[j] <- sum(H.prime2[,j]^expon*row.objects[,2])/weight                      
    }

   print("Computing OncoGPS layout")   
   flush.console()   

   nf <- layout(matrix(1, 1, 1, byrow=T), 1, 1, FALSE)

   x.min <- min(c(xp, row.objects[,1]))
   x.max <- max(c(xp, row.objects[,1]))
   x.len <- x.max - x.min
   x.min <- x.min - x.len/10
   x.max <- x.max + x.len/10
   y.min <- min(c(yp, row.objects[,2]))
   y.max <- max(c(yp, row.objects[,2]))
   y.len <- y.max - y.min
   y.min <- y.min - y.len/9
   y.max <- y.max + y.len/10
   lims <- c(x.min, x.max, y.min, y.max)
   print("lims")
   print(lims)   
   flush.console()      
   kernel.width <- kernel.width*(max(lims) - min(lims))
   print("kernel.width:")
   print(kernel.width)
   print("size.grid:")
   print(size.grid)
   print("Compute state densities")   
   flush.console()   

   Pxy <- array(0, dim= c(size.grid, size.grid, k.classes), dimnames=list(NULL, NULL, all.classes))

   for (k in all.classes) {
       print(paste("k:", k))
       class.size <-  sum(cutree.model == k)
       if (class.size == 0) stop(paste("ERROR: state ", k, " has no samples"))
       print(paste("class size:", sum(cutree.model == k)))
       flush.console()          
       x.sub <- xp[cutree.model == k]
       y.sub <- yp[cutree.model == k]
       kde2d.xy <- kde2d(x.sub, y.sub, n = size.grid, h = c(kernel.width, kernel.width), lims=lims)
       x.coor <-  kde2d.xy$x
       y.coor <-  kde2d.xy$y
       Pxy[,, k] <- kde2d.xy$z
   }

   print("Compute winning state density per grid point")   
   flush.console()   
   winning.class <- matrix(NA, nrow=size.grid, ncol=size.grid)
   final.Pxy <- matrix(0, nrow=size.grid, ncol=size.grid)                       
   for (i in 1:size.grid) {
       for (j in 1:size.grid) {
           winning.class[i, j] <- dimnames(Pxy)[[3]][which.max(Pxy[i, j, ])]
           final.Pxy[i, j] <- Pxy[i, j, winning.class[i, j]]
       }
   }
   final.Pxy <- (final.Pxy - min(final.Pxy))/(max(final.Pxy) - min(final.Pxy))

  # Make mask for outside of the convex hull

   print("Make mask for outside of the convex hull")
   flush.console()   

      mask <- matrix(0, nrow=size.grid, ncol=size.grid)
      for (i in 1:size.grid) {
         for (j in 1:size.grid) {
            x.p <- x.coor[i]
            y.p <- y.coor[j]
            if (inside.owin(x.p, y.p, w)) mask[i, j] <- 1
         }
      }
   print("Making layout plots")
   flush.console()   

# Show incremental generation of Onco-GPS map
   
#  Plot nodes

   par(mar=c(2,3,1,3))
   plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min , 1.1*y.max), bty="n", axes=F, xlab="", ylab="")               
   for (i in 1:nrow(row.objects)) {
       pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
       text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
   }
   points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)

#  Plot nodes and triangles

   par(mar=c(2,3,1,3))
   plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min , 1.1*y.max), bty="n", axes=F, xlab="", ylab="")               
   for (i in 1:nrow(row.objects)) {
       pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
       text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
   }
   points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)
    for (tri in 1:length(triangles)) {
        triangle.nodes.x  <- c(row.objects[triangles[[tri]][1], 1], row.objects[triangles[[tri]][2], 1], row.objects[triangles[[tri]][3], 1],
                               row.objects[triangles[[tri]][1], 1])
        triangle.nodes.y  <- c(row.objects[triangles[[tri]][1], 2], row.objects[triangles[[tri]][2], 2], row.objects[triangles[[tri]][3], 2],
                               row.objects[triangles[[tri]][1], 2])
        points(triangle.nodes.x, triangle.nodes.y, type="l", col="black", lwd=1, cex=1.25)              
    }

 # Plot nodes, triangles and samples
   
   par(mar=c(2,3,1,3))
   plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min , 1.1*y.max), bty="n", axes=F, xlab="", ylab="")               
    for (tri in 1:length(triangles)) {
        triangle.nodes.x  <- c(row.objects[triangles[[tri]][1], 1], row.objects[triangles[[tri]][2], 1], row.objects[triangles[[tri]][3], 1],
                               row.objects[triangles[[tri]][1], 1])
        triangle.nodes.y  <- c(row.objects[triangles[[tri]][1], 2], row.objects[triangles[[tri]][2], 2], row.objects[triangles[[tri]][3], 2],
                               row.objects[triangles[[tri]][1], 2])
        points(triangle.nodes.x, triangle.nodes.y, type="l", col="black", lwd=1, cex=1.25)              
    }

    for (i in 1:length(xp)) {
        col <- categ.col2[match(cutree.model[i], all.classes)]
        points(xp[i], yp[i], col="black", bg=col, pch=21, cex=point.cex)
    }

   for (i in 1:nrow(row.objects)) {
       pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
       text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
   }
   points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)

#  full Onco-GPS   
   
   par(mar=c(2,3,1,3))
   plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min , 1.1*y.max), bty="n", axes=F, xlab="", ylab="")               

   for (i in 1:size.grid) {
       for (j in 1:size.grid) {
           x.p <- x.coor[i]
           y.p <- y.coor[j]
           col <- mycol.class[ceiling((myncolors - 1) * final.Pxy[i, j] + 1), match(winning.class[i, j], all.classes)]
           points(x.p, y.p, col=col, pch=15, cex=1)
       }
   }
   levels <- seq(0, 1, 1/contour.levels)
   lc <- contourLines(x=x.coor, y=y.coor, z=final.Pxy, levels=levels)
   for (i in 1:length(lc)) points(lc[[i]]$x, lc[[i]]$y, type="l", col= brewer.pal(9, "Blues")[contour.tone], lwd=1)      
      
   for (i in 1:size.grid) {
       for (j in 1:size.grid) {
           x.p <- x.coor[i]
           y.p <- y.coor[j]
           if (mask[i,j] == 0) {
              points(x.p, y.p, col="white", pch=15, cex=1)
           }
        }
    }
      
    text(x.min + 0.05*x.len, 1.08*y.max, "OncoGenic Positional System (Onco-GPS) Map",
              cex= 1.35, family="Times", pos=4, font=4, col="darkblue") # fontface="italic", 
    text(x.min + 0.05*x.len, 1.02*y.max, paste("Basic Layout: Samples (", length(xp),
              ") and States (", k.classes, ")", sep=""), cex = 1.1, font=2, family="", pos=4, col="darkred")
    text(x.min, y.min, description, cex = 1, font=2, family="", pos=4, col="darkblue")

    for (tri in 1:length(triangles)) {
        triangle.nodes.x  <- c(row.objects[triangles[[tri]][1], 1], row.objects[triangles[[tri]][2], 1], row.objects[triangles[[tri]][3], 1],
                               row.objects[triangles[[tri]][1], 1])
        triangle.nodes.y  <- c(row.objects[triangles[[tri]][1], 2], row.objects[triangles[[tri]][2], 2], row.objects[triangles[[tri]][3], 2],
                               row.objects[triangles[[tri]][1], 2])
        points(triangle.nodes.x, triangle.nodes.y, type="l", col="black", lwd=1, cex=1.25)              
    }
   
    for (i in 1:length(xp)) {
        col <- categ.col2[match(cutree.model[i], all.classes)]
        points(xp[i], yp[i], col="black", bg=col, pch=21, cex=point.cex)
    }

   for (i in 1:nrow(row.objects)) {
       pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
       text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
   }
   points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)

   leg.txt <- all.classes
   pch.vec <- rep(21, length(leg.txt))
   col <- unique(categ.col2[match(cutree.model, all.classes)])
                        
   legend(x.max - 0.04*x.len, 1.1*y.max, legend=leg.txt, bty="n", xjust=0, yjust= 1, pch = pch.vec, title="States",   
          pt.bg = col, col = "black", cex = 1, pt.cex = 1.5)

   if (plot_sample_names == T) pointLabel(xp, yp, labels=colnames(m.1), cex=0.85, col="darkgreen")

   save(contour.levels, contour.tone, all.classes, svm.mod, expon, description, Ns.1, norm.thres, triangles, nodes, node.nicknames,
        sample.names.1, k.classes, cutree.model, size.grid, xp, yp, x.coor, y.coor, row.objects, mycol, categ.col2,
        X, mask, x.min, x.max, low.range.color, high.range.color, point.cex, nodes.mean, nodes.sd, nodes.min, nodes.max,
        x.len, y.min, y.max, y.len, binary.col, max.cont.color, mycol.class, myncolors, final.Pxy, winning.class, file = OPM.objects.file)

   if (!is.null(output.file))  dev.off()
}

#-------------------------------------------------------------------------------------------------
   CCBA_extract_subset_from_dataset.v1 <- function(
   #
   #  Extracts a subset of features and samples and makes a dataset containing only those
   #  P. Tamayo Jan 17, 2016
    
      input.ds,           # Input dataset
      features = "ALL",   # Subset of features (row names) to extract. Default is "ALL"
      samples = "ALL",    # Subset of samples (column names) to extract. Default is "ALL"
      output.ds)           # Output dataset with desired subset
    {
  
   # Read dataset

      dataset <- CCBA_read_GCT_file.v1(filename = input.ds)
      A <- data.matrix(dataset$ds)
      g.names <- dataset$row.names
      g.descs <- dataset$descs
      sample.names <- dataset$names

      print(dim(A))

      if (features == "ALL") {
         row.locs <- seq(1, nrow(A))
      } else {
         row.locs <- match(features, g.names)
         row.locs <- row.locs[!is.na(row.locs)]
         print(row.locs)
      }

      if (samples == "ALL") {
         col.locs <- rep(T, ncol(A))
      } else {
         col.locs <- match(samples, sample.names)
         col.locs <- col.locs[!is.na(col.locs)]         
      }

      B <- A[row.locs, col.locs]
      print(dim(B))
      colnames(B) <- colnames(A)[col.locs]
      row.names(B) <- row.names(A)[row.locs]
        
      CCBA_write.gct.v1(gct.data.frame = as.matrix(B), descs = row.names(B), filename = output.ds)
   }

#-------------------------------------------------------------------------------------------------
   CCBA_IC_selection.v1 <- function(
   #
   # Given a target profile from datatset (ds1) match features from a second dataset (ds2)
   # P. Tamayo Jan 17, 2016
   #
       ds1,                                        # Input dataset with target (CLS or GCT) file
       target.name,                                # Name of target in ds1
       target.combination.op          = "max",     # operation to reduce multiple targets to one vector of values
       cond.feature.name              = NULL,      # Feature in ds1 to be used as conditional variable for CMI. "TISSUE" uses the tissue type
       ds2,                                        # Input feature dataset 
       n.markers                      = 20,        # Number of markers for bootstrap, heatmap and mds plot
       n.perm                         = 3,         # Number of random permutations
       permutation.test.type          = "standard", # balanced  subclass.stratified
       n.boot                         = 50,        # Number of bootstrap samples for confidence interval of the n.markers
       seed                           = 86876,     # Random number generator seed
       assoc.metric.type              = "IC",      # Association metric: IC, RNMI, NMI, SMI, AUC.ROC, AUC.REV, DIFF.MEDIANS, DIFF.MEANS, S2N, T.TEST, CORR
       direction                      = "positive",  # Direction of feature matching
       sort.target                    = TRUE,      # Sort columns according to phenotype (heatmap)
       results.file.pdf,                           # PDF output file
       results.file.txt,                           # TXT output file
       results.file.gct               = NULL,      # GCT file with top results                 
       sort.columns.inside.classes    = T,         # Sort columns in heatmap of top matches inside each target class               
       cluster.top.markers            = F,         # Sort rows in hetamap of top matches (F, "each.class" or "both.classes")                   
       consolidate.identical.features = F,         # Consolidate identical features: F or "identical" or "similar" 
       cons.features.hamming.thres    = 3,         # If consolidate.identical.features = "similar" then consolidate features within this Hamming dist. thres.
       minimum.nonNA.entries          = 3,         # Minimum number of non NA values for every feature
       minimum.distinct.values        = 2,         # Minimum number of distinct values for every feature     
       locs.table.file                = NULL,      # Table with chromosonal locations per feature (gene)
       save.matched.dataset           = F,         # Save target-fetaures matched dataset                                          
       produce.aux.histograms         = F,         # Produce histograms of assoc. metric distribution etc.                         
       produce.heat.map               = T,         # Produce heatmap                                                               
       produce.mds.plots              = T,         # Produce multi-dimensional scaling (mds) plot (Landscape plot)                 
       character.scaling              = 1,         # character scaling for heatmap
       mds.plot.type                  = "smacof",  # mds algorithm
       target.style                   = "color.bar", # "color.bar" or "bar.graph"
       knn                            = 3,         # k for knn-based assoc. metrics
       display.cor.coeff.heatmap      = F,         # show correlation coeff. in heatmap/text file
       n.grid                         = 25,        # grid size for kernel-based assoc. metrics                                     
       phen.table                     = NULL,      # Table with phenotypes for each sample (optional)
       phen.column                    = NULL,      # Column in phen.table containing the relevant phenotype info
       phen.selected                  = NULL,      # Use only samples of these phenotypes in analysis
       debug.mode                     = F,         # Print additional info to diagnose problems
       missing.value.color            = "wheat",   # Missing feature color
       binary.0_value.color           = "lightgray",  # Binary feature's 0's color 
       binary.1_value.color           = "black")   # Binary feature's 1's color 
  {

   set.seed(seed)

   time1 <- proc.time()

   pdf(file=results.file.pdf, height=11, width=8.5)
   if (is.null(results.file.gct)) {
      results.file.gct = paste(results.file.pdf, ".gct", sep="")
   }
     
   # Read table with HUGO gene symbol vs. chr location
   
   if (!is.null(locs.table.file)) {
      locs.table <- read.table(locs.table.file, header=T, sep="\t", skip=0, colClasses = "character")
    }

   # Read input files 

   print("Reading features file...")
   
   if (regexpr(pattern=".gct", ds2) != -1) {
      dataset2 <- CCBA_read_GCT_file.v1(filename = ds2)
      m2 <- data.matrix(dataset2$ds)
      feature.names2 <- dataset2$row.names
      sample.names2 <- colnames(m2)
   } else if (regexpr(pattern=".txt", ds2) != -1) {
      df1 <- read.table(ds2, header=T, row.names=1, sep="\t", skip=0)
      df1 <- data.matrix(df1)
      m2 <- t(df1)
      feature.names2 <- row.names(m2)
      sample.names2 <- colnames(m2)
    }

   # Filter samples with only the selected phenotypes 

   if (!is.null(phen.selected)) {
      print(paste("Subselecting samples with phenotype: ", phen.selected))
      samples.table <- read.delim(phen.table, header=T, row.names=1, sep="\t", skip=0)
      table.sample.names <- row.names(samples.table)
      locs1 <- match(colnames(m2), table.sample.names)
      phenotype <- as.character(samples.table[locs1, phen.column])
      table(phenotype)
      locs2 <- NULL
      for (k in 1:ncol(m2)) {   
         if (!is.na(match(phenotype[k], phen.selected))) {
            locs2 <- c(locs2, k)
         }
      }
      print(paste("Matching phenotype total number of samples:", length(locs2)))
      m2 <- m2[, locs2]
      sample.names2 <- colnames(m2)
      phenotype <- phenotype[locs2]
      print(table(phenotype))
      print(dim(m2))
    }
   
   print("Reading target file...")
   
   if (regexpr(pattern=".gct", ds1) != -1) {
      dataset1 <- CCBA_read_GCT_file.v1(filename = ds1)
      m1 <- data.matrix(dataset1$ds)
#      m1[is.na(m1)] <- 0          
      feature.names1 <- dataset1$row.names
      sample.names1 <- colnames(m1)

   if (length(target.name) > 1) {  # multiple targets => combine them using target.combination.op 
         print("multi target")
         target <- apply(m1[target.name,], MARGIN=2, FUN=target.combination.op)
         target.name <- paste(target.name, collapse="__")
         print(target.name)
      } else { # single target
         print("single target")        
         target <- m1[target.name,]
         print(target.name)
      }

     # exclude samples with target == NA

      print(paste("initial target length:", length(target)))      
      locs <- seq(1, length(target))[!is.na(target)]
      m1 <- m1[,locs]
      target <- target[locs]
      sample.names1 <- sample.names1[locs]

      print(paste("target length after excluding NAs:", length(target)))      

      overlap <- intersect(sample.names1, sample.names2)
      print(paste("Size of overlap:", length(overlap)))
      locs1 <- match(overlap, sample.names1)
      locs2 <- match(overlap, sample.names2)
      m1 <- m1[, locs1]

      m2 <- m2[, locs2]
      target <- target[locs1]
      
      print(paste("final target length:", length(target)))
      
      print(dim(m1))
      print(dim(m2))

      if (!is.null(cond.feature.name)) {
         if (cond.feature.name == "TISSUE") {
            tissue.type <- vector(length=ncol(m1), mode="character")
            for (k in 1:ncol(m1)) {
               temp <- strsplit(colnames(m3)[k], split="_") 
               tissue.type[k] <- paste(temp[[1]][2:length(temp[[1]])], collapse="_")
             }
            cond.feature <- match(tissue.type, unique(tissue.type))
         } else {
            cond.feature <- m1[cond.feature.name,]
         }
       } else {
         cond.feature <- NULL
       }
      
      if (save.matched.dataset == T) {
         CCBA_write.gct.v1(gct.data.frame = m1, descs = row.names(m1), filename = paste(ds1, ".MATCHED.SET.gct", sep=""))
         CCBA_write.gct.v1(gct.data.frame = m2, descs = row.names(m2), filename = paste(ds2, ".MATCHED.SET.gct", sep=""))
         print(paste(ds1, ".MATCHED.SET.gct", sep=""))
         print(paste(ds2, ".MATCHED.SET.gct", sep=""))
      }

   } else if (regexpr(pattern=".txt", ds1) != -1) {
      df1 <- read.table(ds1, header=T, row.names=1, sep="\t", skip=0)
      m1 <- t(df1)
      m1[is.na(m1)] <- 0
      sample.names1 <- colnames(m1)
      overlap <- intersect(sample.names1, sample.names2)
      locs1 <- match(overlap, sample.names1)
      locs2 <- match(overlap, sample.names2)
      m1 <- m1[, locs1]
      m2 <- m2[, locs2]
      if (length(target.name) > 1) {  # multiple targets => combine them using target.combination.op 
         target <- apply(m1[target.name,], MARGIN=2, FUN=target.combination.op)
         target.name <- paste(target.name, collapse="__")
      } else { # single target
         target <- m1[target.name,]
      }
      classes <- unique(target)
      target <- match(target, classes)
      print(dim(m1))
      print(dim(m2))
                        
   } else if (regexpr(pattern=".cls", ds1) != -1) {
      CLS <- CCBA_ReadClsFile(ds1)
      if (is.numeric(CLS$class.list)) {
         target <- as.numeric(CLS$class.list)
       } else {
         target <- match(CLS$class.list, rev(unique(CLS$class.list)))
       }
   }
  target <- as.numeric(target)

  if (debug.mode == T) {
     print("target:")
     print(target)
     print(table(target))
   }
   
  # Filter out features withless than minimum.nonNA.entries non-NA vals or not enough
  # distinct values (minimum.distinct.values)

   num.dif <- num.noNAs <- rep(0, nrow(m2))
   for (j in 1:nrow(m2)) {
      x <- m2[j, !is.na(m2[j,])]
      num.noNAs[j] <- length(x)
      num.dif[j] <- length(unique(x))
   }
   m2 <- m2[(num.noNAs >= minimum.nonNA.entries) & (num.dif >= minimum.distinct.values),]

   #  Consolidate identical features

   if (consolidate.identical.features == "identical") {

       # This is a very fast way to eliminate perfectly identical features compared
       # with what we do below in "similar"
      print(paste("Consolidating features..."))
      summary.vectors <- apply(m2, MARGIN=1, FUN=paste, collapse="")
      ind <- order(summary.vectors)
      summary.vectors <- summary.vectors[ind]
      m2 <- m2[ind,]
      taken <- i.count <- rep(0, length(summary.vectors))
      i <- 1
      while (i <= length(summary.vectors)) {
         j <- i + 1
         while ((summary.vectors[i] == summary.vectors[j]) & (j <= length(summary.vectors))) {
           j <- j + 1
         }
        i.count[i] <- j - i
        if (i.count[i] > 1) taken[seq(i + 1, j - 1)] <- 1
        i <- j
      }
      if (sum(i.count) != length(summary.vectors)) stop("ERROR")    # add counts in parenthesis
      i.count <- ifelse(i.count > 1, paste("(", i.count, ")", sep=""), rep(" ", length(i.count)))
      row.names(m2) <- paste(row.names(m2), i.count)
      m2 <- m2[taken == 0,]

      # this uses the hamming distance to consolidate similar features up to the Hamming dist. threshold
      
    } else if (consolidate.identical.features == "similar") { 
   print(paste("Consolidating features..."))
   hamming.matrix <- hamming.distance(m2)
   taken <- rep(0, nrow(m2))
   for (i in 1:nrow(m2)) {
    if (taken[i] == 0) { 
       similar.features <- row.names(m2)[hamming.matrix[i,] <= cons.features.hamming.thres]
       if (length(similar.features) > 1) {
           row.names(m2)[i]  <- paste(row.names(m2)[i], " (", length(similar.features), ")", sep="")  # add counts in brackets
           locs <- match(similar.features, row.names(m2))
           taken[locs] <- 1
           taken[i] <- 0
        }
      }
   }
  m2 <- m2[taken == 0,]
 }
   print(dim(m2))
   
   # Add location info

   if (!is.null(locs.table.file)) {
      print(paste("Adding location info..."))
      gene.symbol <- row.names(m2)
      chr <- rep(" ", length(gene.symbol))
      for (i in 1:length(gene.symbol)) {
        temp1 <- strsplit(gene.symbol[i], split="_")
        temp2 <- strsplit(temp1[[1]][1], split="\\.")
        gene.symbol[i] <- ifelse(temp2[[1]][1] == "", temp1[[1]][1], temp2[[1]][1])
        loc <- match(gene.symbol[i], locs.table[,"Approved.Symbol"])
        chr[i] <- ifelse(!is.na(loc), paste("(", locs.table[loc, "Chromosome"], ") ", sep=""), " ")
       }
      row.names(m2)  <- paste(row.names(m2), chr)
      print(paste("Total unmatched to chromosomal locations:", sum(chr == " "), "out of ", nrow(m2), "features"))
    }
   
   if (sort.target == TRUE) {
      ind <- order(target, decreasing=T)
      target <-  target[ind]
      if (!is.null(cond.feature.name)) {
         cond.feature <-  cond.feature[ind]
      }
      m3 <- m2[, ind]
   } else {
      m3 <- m2
    }

   N <- ncol(m3)
   p <- nrow(m3)
   if (direction == "negative") {
      if (length(table(target)) > N*0.5) { # continuous target
         target2 <- -target
      } else {
         target2 <-  1 - target
      }
   } else if (direction == "positive") {
      target2 <- target
   } else {
      stop(paste("Unknown direction:", direction))
   }

   target2 <- target2 + 10 * .Machine$double.eps * rnorm(length(target2))
   print(paste("Length of target:", length(target2)))
      
   # Find association of target vs. features using selected metric

   print("Finding association of target vs. features using selected metric...")

   metric <- cor.val <- vector(mode="numeric", length=p)
   metric.rand <- matrix(0, nrow=p, ncol=n.perm)

   target.rand <- matrix(target2, nrow=n.perm, ncol=N, byrow=TRUE)
   if (permutation.test.type == "standard") {    # balanced  subclass.stratified
      for (i in 1:n.perm) target.rand[i,] <- sample(target.rand[i,])
   } else if (permutation.test.type == "balanced") {    # balanced  subclass.stratified

   # we removed this option because of too much noise
     
   } else if (permutation.test.type == "subclass.stratified") {
      subclass.type <- vector(length=ncol(m3), mode="character")
      for (k in 1:ncol(m3)) {
         temp <- strsplit(colnames(m3)[k], split="_") 
         subclass.type[k] <- paste(temp[[1]][2:length(temp[[1]])], collapse="_")
      }
      all.types <- unique(subclass.type)
      for (k in 1: length(all.types)) {
         V2 <- as.matrix(target.rand[, all.types[k] == subclass.type])
         if (ncol(V2) > 1) for (i in 1:n.perm) V2[i,] <- sample(V2[i,])
         target.rand[, all.types[k] == subclass.type] <- V2
       }
    } else {
      stop(paste("Unknown permutation test type:", permutation.test.type))
    }

   for (i in 1:p) {
      feature <- m3[i,]
      metric[i] <- CCBA_assoc.metric.v1(target = target2, feature = feature, cond.feature = cond.feature, type=assoc.metric.type, knn=knn, n.grid=n.grid)
      cor.val[i] <- cor(target2, feature)
      if (i %% 100 == 0) print(paste(" feature #", i, " out of ", p, "metric:", metric[i]))
      for (k in 1:n.perm) {
         metric.rand[i, k] <- CCBA_assoc.metric.v1(target = target.rand[k,], feature = feature, cond.feature = cond.feature, type=assoc.metric.type,
                                           knn=knn, n.grid=n.grid)
       }
    }

   # Sort features

   ind <- order(metric, decreasing=T)
   metric <- metric[ind]
   cor.val <- cor.val[ind]
   metric.rand <- metric.rand[ind,]
   metric.rand.max <- apply(metric.rand, MARGIN=1, FUN=max)
   m3 <- m3[ind,]
   feature.names2 <- row.names(m3) 
 
   n.markers <- ifelse(nrow(m3) < 2*n.markers, floor(nrow(m3)/2), n.markers)
   
   # Compute Confidence Intervals for top features using 0.632 bootstrap

   if (!is.null(n.boot)) {
      print("computing bootstrap confidence intervals...")
      b.size <- ceiling(0.632*N)
      metric.CI <- matrix(0, nrow=p, ncol=3)
      boots.null <- matrix(0, nrow=2*n.markers, ncol=n.boot)
      for (k in 1:n.boot) {
         locs <- sample(seq(1, N), b.size, replace=T)
         m3.sample <- m3[c(seq(1, n.markers, 1), seq(p, p - n.markers + 1, -1)), locs]
         target2.sample <- target2[locs]
         for (i in 1:(2*n.markers)) {
            feature.sample <- as.numeric(m3.sample[i,])
            boots.null[i, k] <- CCBA_assoc.metric.v1(target = target2.sample, feature = feature.sample,
                                             cond.feature = cond.feature, type=assoc.metric.type, knn=knn, n.grid=n.grid)
          }
       }
       
       for (i in 1:n.markers) metric.CI[i, ] <- quantile(boots.null[i,], probs = c(0.05, 0.5, 0.95), na.rm = T)
       j <- n.markers + 1
       for (i in seq(p, p - n.markers + 1, -1)) {
          metric.CI[i, ] <- quantile(boots.null[j,], probs = c(0.05, 0.5, 0.95))
          j <- j+1
       }
    } else {
      metric.CI <- matrix(0, nrow=p, ncol=3)
    }
  
   # Make histogram and QQplot

   if (produce.aux.histograms == T) {
      nf <- layout(matrix(c(1, 2), 2, 1, byrow=T), c(1, 1), 1, FALSE)
      h <- hist(metric.rand, breaks=40, col="steelblue", main="Global Null Dist", xlab=paste("Metric:", assoc.metric.type),
                ylab = "P(M)", xlim=range(metric))
      for (i in 1:p) lines(c(metric[i], metric[i]), c(0, -0.025*max(h$counts)), lwd=1, col="black")
      chart.QQPlot(metric, main = paste("QQ Plot for ", assoc.metric.type, " Association Metric"), distribution = 'norm',
                   envelope=0.95, pch=20, cex=0.6, lwd=3)
    }

   # Compute p-values and FDR

   print("computing p-vals and FDRs...")
   
   p.val1 <- p.val2 <- FDR1 <- FDR1.lower <- FDR2 <- FDR2.lower <- FDR3 <- FDR3.lower <- FWER.p.val <- Bonfe.p.val <- rep(0, p)
   p.val1.tag <- p.val2.tag <- p.val3.tag <- FDR1.tag <- FDR2.tag <- FDR3.tag <- FWER.p.val.tag <- Bonfe.p.val.tag <- rep(0, p)

   for (i in 1:p) {
      p.val1[i] <- sum(metric.rand[i,] > metric[i])/n.perm
      p.val2[i] <- sum(metric.rand > metric[i])/(p*n.perm)
      FWER.p.val[i] <- sum(metric.rand.max > metric[i])/p
   }
   if (assoc.metric.type == "DIFF.MEANS") { # use local p-vals: p.val1
     FDR1 <- p.adjust(p.val1, method = "fdr", n = length(p.val1))
     FDR1.lower <- p.adjust(1 - p.val1, method = "fdr", n = length(p.val1))
      FDR2 <- FDR2.lower <- rep(1, length(p.val1))
   } else { # Use p.val2 for the other metrics (have global null distributions)
     print("computing FDRs using p.adjust...")
     FDR1 <- p.adjust(p.val2, method = "fdr", n = length(p.val2))
     FDR1.lower <- p.adjust(1 - p.val2, method = "fdr", n = length(p.val2))
     print("computing FDRs using q.values...")     
     FDR2 <- FDR1    # qvalue crashes sometimes: use FDR1 
     FDR2.lower <- FDR1.lower
   }

   for (i in 1:p) {
      FDR3[i] <- (sum(metric.rand >= metric[i])/(p*n.perm))/(sum(metric >= metric[i])/p)
      FDR3.lower[i] <- (sum(metric.rand <= metric[i])/(p*n.perm))/(sum(metric <= metric[i])/p)
    }
   for (i in 1:p) {
     FDR3[i] <- min(FDR3[i:p]) 
     FDR3.lower[i] <- min(FDR3[1:i])
   }

   lower.side <- rep(0, p)
   for (i in 1:p) {
      if (assoc.metric.type == "AUC.ROC") {       
        if (metric[i] < 0.5) lower.side[i] <- 1
      } else { # RNMI, NMI, DIFF.MEDIANS, DIFF.MEANS, S2N, T.TEST
        if (metric[i] < 0) lower.side[i] <- 1
      }
    }
   
   for (i in 1:p) {
      if (lower.side[i] == 1) {
         p.val1[i] <- 1 - p.val1[i]
         p.val2[i] <- 1 - p.val2[i]
         FWER.p.val[i] <- 1 - FWER.p.val[i]
         Bonfe.p.val[i] <- 1 - Bonfe.p.val[i]
         FDR1[i] <- FDR1.lower[i]
         FDR2[i] <- FDR2.lower[i]
         FDR3[i] <- FDR3.lower[i]
       }
      if (p.val1[i] == 0) {
          p.val1[i] <- 1/n.perm
          p.val1.tag[i] <- "<"
      }
      if (p.val2[i] == 0) {
          p.val2[i] <- 1/(p*n.perm)
          p.val2.tag[i] <- "<"
      }
      if (FDR3[i] == 0) {
          FDR3[i] <- (1/(p*n.perm))/(i/p)
          FDR3.tag[i] <- "<"
      }
      if (FWER.p.val[i] == 0) {
          FWER.p.val[i] <- 1/p
          FWER.p.val.tag[i] <- "<"
      }
    }
   for (i in 1:p) Bonfe.p.val[i] <- ifelse(p.val2[i] * p > 1, 1, p.val2[i] * p)

   # Make histograms of p-values and FDRs

   if (produce.aux.histograms == T) {
     
      nf <- layout(matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), 3, 3, byrow=T), c(1, 1, 1), c(1, 1, 1), FALSE)
      h <- hist(p.val1, breaks=40, col="darkblue", main="p-val1", xlab="p-value", ylab = "P(p-val1)")
      h <- hist(p.val2, breaks=40, col="darkblue", main="p-val2", xlab="p-value", ylab = "P(p-va2l)")
      h <- hist(FDR1, breaks=40, col="darkblue", main="FDR1", xlab="FDR (HB)", ylab = "P(FDR1)")
      h <- hist(FDR2, breaks=40, col="darkblue", main="FDR2", xlab="FDR (Storey)", ylab = "P(FDR2)")
      h <- hist(FDR3, breaks=40, col="darkblue", main="FDR3", xlab="FDR (Theor.)", ylab = "P(FDR3)")
      h <- hist(FWER.p.val, breaks=40, col="darkblue", main="FWER p-values", xlab="p-value", ylab = "P(FWER p-val)")
      h <- hist(Bonfe.p.val, breaks=40, col="darkblue", main="Bonferroni p-values", xlab="p-value", ylab = "P(Bonfe p-val)")
    }

   metric.CI <- signif(metric.CI, 4)
   metric.CI[metric.CI == 0] <- "-"                      
   if (display.cor.coeff.heatmap == T) {
      report1 <- cbind(seq(1, p), feature.names2, signif(metric, 3), metric.CI,
                    paste(p.val1.tag, signif(p.val1, 3), sep=""),
                    paste(p.val2.tag, signif(p.val2, 3), sep=""),
                    paste(FDR1.tag, signif(FDR1, 3), sep=""),
                    paste(FDR2.tag, signif(FDR2, 3), sep=""),
                    paste(FDR3.tag, signif(FDR3, 3), sep=""),
                    paste(FWER.p.val.tag, signif(FWER.p.val, 3), sep=""),
                    paste(Bonfe.p.val.tag, signif(Bonfe.p.val, 3), sep=""),
                    signif(cor.val), 3)
       colnames(report1) <- c("Rank", "Feature", assoc.metric.type, "5% CI", "50% CI", "95% CI", "p-val1 (local)", "p-val2 (global)", "FDR1",
                          "FDR2", "FDR3", "FWER p-val", "Bonfe p-val", "Corr. Coef.")
     } else {
      report1 <- cbind(seq(1, p), feature.names2, signif(metric, 3), metric.CI,
                    paste(p.val1.tag, signif(p.val1, 3), sep=""),
                    paste(p.val2.tag, signif(p.val2, 3), sep=""),
                    paste(FDR1.tag, signif(FDR1, 3), sep=""),
                    paste(FDR2.tag, signif(FDR2, 3), sep=""),
                    paste(FDR3.tag, signif(FDR3, 3), sep=""),
                    paste(FWER.p.val.tag, signif(FWER.p.val, 3), sep=""),
                    paste(Bonfe.p.val.tag, signif(Bonfe.p.val, 3), sep=""))
       colnames(report1) <- c("Rank", "Feature", assoc.metric.type, "5% CI", "50% CI", "95% CI", "p-val1 (local)", "p-val2 (global)", "FDR1",
                          "FDR2", "FDR3", "FWER p-val", "Bonfe p-val")
     }
     
   print(noquote(report1[1:n.markers,]))
   print(noquote(report1[seq(p, p - n.markers + 1, -1),]))

   write.table(report1, file=results.file.txt, quote=F, col.names = T, row.names = F, append = F, sep="\t")

   metric.sorted <- metric[c(1:n.markers, seq(p - n.markers + 1, p))]
   cor.val.sorted <- cor.val[c(1:n.markers, seq(p - n.markers + 1, p))]
   p.val1.sorted <- p.val1[c(1:n.markers, seq(p - n.markers + 1, p))]
   p.val2.sorted <- p.val2[c(1:n.markers, seq(p - n.markers + 1, p))]
   FDR1.sorted <- FDR1[c(1:n.markers, seq(p - n.markers + 1, p))]

   # Make heatmap of top and bottom features

   if (produce.heat.map == T) {
      print("making heatmap...")

   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   max.cont.color <- 512
   mycol <- c(mycol,
              missing.value.color,                  # Missing feature color
              binary.0_value.color,                 # Binary feature's 0's color 
              binary.1_value.color)                 # Binary feature's 1's color 

      
      nf <- layout(matrix(c(1, 2), 2, 1, byrow=T), 1, c(1, 11), FALSE)

      m.V <- m3[c(1:n.markers, seq(p - n.markers + 1, p)),]
      target.V <- target

       # sort columns inside classes if target is not continuous
      
      if (sort.columns.inside.classes == T & length(table(target.V)) < ncol(m.V)*0.5)  {  
         num.phen <- length(unique(target.V))
         for (k in unique(target.V)) {
            V3 <- m.V[, target.V == k]
            cl <- target.V[target.V == k]
            c.names <- colnames(m.V)[target.V == k]

            V3.noNAs <- V3
            V3.noNAs[is.na(V3.noNAs)] <- 0
            dist.matrix <- dist(t(V3.noNAs))  
            s <- smacofSym(dist.matrix, ndim=1)
            ind <- order(s$conf, decreasing=T)
            
            V3 <- V3[, ind]
            cl <- cl[ind]
            c.names <- c.names[ind]
            target.V[target.V == k] <- cl
            m.V[, target.V == k] <- V3
            colnames(m.V)[target.V == k] <- c.names
         }
      }

      if (cluster.top.markers == "each.class") {
         marker.ind <- c(rep(1, n.markers), rep(2, n.markers))
         num.phen <- length(unique(marker.ind))
         for (k in unique(marker.ind)) {
            V3 <- m.V[marker.ind == k,]
            row.names.V3 <- row.names(m.V)[marker.ind == k]
            metric.sorted.V <- metric.sorted[marker.ind == k]
            cor.val.sorted.V <-cor.val.sorted[marker.ind == k]            
            p.val1.sorted.V <- p.val1.sorted[marker.ind == k]
            p.val2.sorted.V <- p.val2.sorted[marker.ind == k]
            FDR1.sorted.V <- FDR1.sorted[marker.ind == k]

            V3.noNAs <- V3
            V3.noNAs[is.na(V3.noNAs)] <- 0
            
            dist.matrix <- dist(V3.noNAs)  
            s <- smacofSym(dist.matrix, ndim=1)
            ind <- order(s$conf, decreasing=T)

            V3 <- V3[ind,]
            row.names.V3 <- row.names.V3[ind]
            metric.sorted.V <- metric.sorted.V[ind]
            cor.val.sorted.V <- cor.val.sorted.V[ind]            
            p.val1.sorted.V <- p.val1.sorted.V[ind]
            p.val2.sorted.V <- p.val2.sorted.V[ind]
            FDR1.sorted.V <-   FDR1.sorted.V[ind]

            m.V[marker.ind == k,] <- V3
            row.names(m.V)[marker.ind == k] <- row.names.V3            
            metric.sorted[marker.ind == k] <- metric.sorted.V
            cor.val.sorted[marker.ind == k] <- cor.val.sorted.V            
            p.val1.sorted[marker.ind == k] <- p.val1.sorted.V
            p.val2.sorted[marker.ind == k] <- p.val2.sorted.V
            FDR1.sorted[marker.ind == k] <-   FDR1.sorted.V
         }
         if (metric.sorted[1] < 0 ) {
            m.V <- apply(m.V, MARGIN=2, FUN=rev)
            metric.sorted <- rev(metric.sorted )
            cor.val.sorted <- rev(cor.val.sorted )            
            p.val1.sorted <- rev(p.val1.sorted)
            p.val2.sorted <- rev(p.val2.sorted)
            FDR1.sorted <- rev(FDR1.sorted)
         }            
   } else if (cluster.top.markers == "both.classes") {

      V3.noNAs <- m.V
      V3.noNAs[is.na(V3.noNAs)] <- 0
      dist.matrix <- dist(V3.noNAs) 
      s <- smacofSym(dist.matrix, ndim=1)
      ind <- order(s$conf, decreasing=T)

      m.V <- m.V[ind,]
      metric.sorted <- metric.sorted[ind]
      cor.val.sorted <- cor.val.sorted[ind]      
      p.val1.sorted <- p.val1.sorted[ind]
      p.val2.sorted <- p.val2.sorted[ind]
      FDR1.sorted <- FDR1.sorted[ind]
      if (metric.sorted[1] < 0 ) {
         m.V <- apply(m.V, MARGIN=2, FUN=rev)
         metric.sorted <- rev(metric.sorted )
         cor.val.sorted <- rev(cor.val.sorted )         
         p.val1.sorted <- rev(p.val1.sorted)
         p.val2.sorted <- rev(p.val2.sorted)
         FDR1.sorted <- rev(FDR1.sorted)
      }            
   }
      cutoff <- 2.5
      x <- as.numeric(target.V)         
      x <- (x - mean(x))/sd(x)         
      ind1 <- which(x > cutoff)
      ind2 <- which(x < -cutoff)
      x[ind1] <- cutoff
      x[ind2] <- -cutoff
      V1 <- ceiling(max.cont.color * (x + cutoff)/(cutoff*2))

       if (target.style == "color.bar") {
           par(mar = c(1, 14, 2, 10))
           image(1:N, 1:1, as.matrix(V1), zlim = c(0, max.cont.color), col=mycol[1: max.cont.color],
                 axes=FALSE, main="", sub = "", xlab= "", ylab="")
           axis(2, at=1:1, labels=paste(target.name, "  "), adj= 0.5, tick=FALSE,
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
           ref <- 1
       } else if (target.style == "bar.graph") {
           par(mar = c(0, 14, 2, 10))
           V1.vec <- as.vector(V1)
           V1.vec <- (V1.vec - min(V1.vec))/(max(V1.vec) - min(V1.vec))

            barplot(V1.vec, xaxs="i", ylim=c(-0.25, 1), col="darkgrey", border="darkgrey", xaxt='n', yaxt='n',
                    ann=FALSE, bty='n', width=1, space=0)
           points(seq(1, length(V1.vec)), V1.vec, type="l", col=1)
           points(rep(0, 11), seq(0, 1, 0.1), type="l", col=1)
           points(seq(1, length(V1.vec)), rep(0, length(V1.vec)), type="l", col=1)
           
           axis(2, at=0.5, labels=paste(target.name, "  "), adj= 0.5, tick=FALSE,
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
           ref <- -0.25
       } else {
           stop(paste("ERROR: unknown target style", target.style))
       }
      
      if (assoc.metric.type == "DIFF.MEANS") { # use local p-vals: p.val1
          if (display.cor.coeff.heatmap == T) {
            axis(4, at=1:1, labels=paste(assoc.metric.type, "p-val (loc)", "FDR", "cor"), adj= 0.5, tick=FALSE,
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
          } else {
            axis(4, at=1:1, labels=paste(assoc.metric.type, "p-val (loc)", "FDR"), adj= 0.5, tick=FALSE,
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
          }
      } else {
          if (display.cor.coeff.heatmap == T) {
             axis(4, at=1:1, labels=paste(assoc.metric.type, "p-val (glob)", "FDR", "cor"), adj= 0.5, tick=FALSE,
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
          } else {
            axis(4, at=ref, labels=paste(assoc.metric.type, "p-val (glob)", "FDR"), adj= 0.5, tick=FALSE,              
                 las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
          }
      }
      V <- m.V

   cutoff <- 2.5
   for (i in 1:nrow(V)) {
      x <- V[i,]                
      locs.non.na <- !is.na(x)
      x.nonzero <- x[locs.non.na]
      x.nonzero2 <- (x.nonzero - mean(x.nonzero))/sd(x.nonzero)         
      x.nonzero2[x.nonzero2 > cutoff] <- cutoff
      x.nonzero2[x.nonzero2 < - cutoff] <- - cutoff      
      s <- strsplit(row.names(V)[i], "_")[[1]]
      suffix <- s[length(s)]

      if (suffix == "MUT" | suffix == "AMP" | suffix == "DEL" | suffix == "AMP_2" | suffix == "AMP_3" | suffix == "DEL_2" | suffix == "DEL_3" |
          suffix == "all" | length(table(x.nonzero)) == 2) {  # Binary feature
         V[i,locs.non.na] <- x.nonzero + max.cont.color + 2   # binary feature colors
       } else {
         V[i, locs.non.na] <- x.nonzero2
         V[i, locs.non.na] <- ceiling(max.cont.color * (V[i,locs.non.na] + cutoff)/(2*cutoff))
         V[i, locs.non.na] <- ifelse (V[i, locs.non.na] > max.cont.color, max.cont.color, V[i, locs.non.na])
       }
      V[i, is.na(x)] <- max.cont.color + 1  # missing feature color 
    }

      V <- apply(V, MARGIN=2, FUN=rev)
      par(mar = c(8, 14, 1, 10))
      image(1:dim(V)[2], 1:dim(V)[1], t(V), zlim = c(0, max.cont.color + 3), col=mycol, axes=FALSE, main="", sub = "", xlab= "", ylab="")
      axis(2, at=1:dim(V)[1], labels=row.names(V), adj= 0.5, tick=FALSE,
           las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
      if (assoc.metric.type == "DIFF.MEANS") { # use local p-vals: p.val1
           if (display.cor.coeff.heatmap == T) {
              mi <- paste(signif(metric.sorted, 3), signif(p.val1.sorted, 3), signif(FDR1.sorted, 3), signif(cor.val.sorted, 3), sep="   ")
            } else {
              mi <- paste(signif(metric.sorted, 3), signif(p.val1.sorted, 3), signif(FDR1.sorted, 3), sep="   ")
            }
      } else {
           if (display.cor.coeff.heatmap == T) {
              mi <- paste(signif(metric.sorted, 3), signif(p.val2.sorted, 3), signif(FDR1.sorted, 3), signif(cor.val.sorted, 3), sep="   ")
           } else {
              mi <- paste(signif(metric.sorted, 3), signif(p.val2.sorted, 3), signif(FDR1.sorted, 3), sep="   ")
           }
      }
      axis(4, at=1:dim(V)[1], labels=rev(mi), adj= 0.5, tick=FALSE,
           las = 1, cex=1, cex.axis=0.65*character.scaling, font.axis=1, line=-1)
      axis(1, at=1:dim(V)[2], labels=colnames(V), adj= 0.5, tick=FALSE,
           las = 3, cex=1, cex.axis=0.3*character.scaling, font.axis=1, line=-1)

      # save top marker dataset
      

       ds <- rbind(target.V, m.V)
       row.names(ds) <- c(target.name, row.names(m.V))
       CCBA_write.gct.v1(gct.data.frame = ds, descs = row.names(ds), filename = results.file.gct)
      
    }

    # Make MDS top features projection
   
  if (produce.mds.plots == T && n.markers > 3) {

   nf <- layout(matrix(1, 1, byrow=T), 1, 1, FALSE)

   total.points <- n.markers
   V2 <- m3[1:total.points,]
   V2[is.na(V2)] <- 0
   
   row.names(V2) <- row.names(m3)[1:total.points]
   metric.matrix <- matrix(0, nrow=nrow(V2), ncol=nrow(V2))
   row.names(metric.matrix)  <- row.names(V2)
   colnames(metric.matrix) <- row.names(V2)
   MI.ref <- metric.sorted[1:total.points]
   for (i in 1:nrow(V2)) {
      for (j in 1:i) {
          metric.matrix[i, j] <- CCBA_assoc.metric.v1(target = V2[j,], feature = V2[i,], type="IC", knn=knn, n.grid=n.grid)
      }
   }

   metric.matrix <- metric.matrix + t(metric.matrix)

   alpha <- 8
   metric.matrix2 <- 1 - ((1/(1+exp(-alpha*metric.matrix))))
   for (i in 1:nrow(metric.matrix2)) metric.matrix2[i, i] <- 0

   smacof.map <- smacofSphere(metric.matrix2, ndim = 2, weightmat = NULL, 
                          ties = "primary", verbose = FALSE, modulus = 1, itmax = 1000, eps = 1e-6)
   x0 <- smacof.map$conf[-1,1]
   y0 <- smacof.map$conf[-1,2]
   r <- sqrt(x0*x0 + y0*y0)
   radius <-  1 - ((1/(1+exp(-alpha*MI.ref))))
   x <- x0*radius/r
   y <- y0*radius/r
   angles <- atan2(y0, x0)
   par(mar = c(10, 2, 10, 2))
   plot(x, y, pch=20, bty="n", xaxt='n', axes = FALSE, type="n", xlab="", ylab="",
        xlim=1.2*c(-max(radius), max(radius)), ylim=1.2*c(-max(radius), max(radius)))
   line.angle <- seq(0, 2*pi-0.001, 0.001)
   for (i in 1:length(x)) {
      line.max.x <- radius[i] * cos(line.angle)
      line.max.y <- radius[i] * sin(line.angle)
      points(line.max.x, line.max.y, type="l", col="gray80", lwd=1)
      points(c(0, x[i]), c(0, y[i]), type="l", col="gray80", lwd=1)
   }
   line.max.x <- 1.2*max(radius) * cos(line.angle)
   line.max.y <- 1.2*max(radius) * sin(line.angle)
   points(line.max.x, line.max.y, type="l", col="purple", lwd=2)
   points(x, y, pch=21, bg="steelblue", col="darkblue", cex=1.1)
   points(0, 0, pch=20, col="red", cex=2.5)
   text(0, 0, labels=target.name, cex=1, col="red", pos=1)

   d <- density(angles, adjust=0.10, n=4000, from= -2*pi, to=2*pi)
   dx <- d$x[1001:3000]
   dd <- d$y[1001:3000]
   dd[1:1000] <- dd[1:1000] + d$y[3001:4000]
   dd[1001:2000] <- dd[1001:2000] + d$y[1:1000]

   xd <- 1.2*max(radius)*cos(dx)
   yd <- 1.2*max(radius)*sin(dx)

   ddd <- (dd - min(dd))/(max(dd) - min(dd))
   ring.colors <- rgb(1, 1 - ddd, 1, maxColorValue = 1)

   for (i in 1:length(xd)) points(xd[i], yd[i], type="p", pch=21, cex=2, bg=ring.colors[i], col=ring.colors[i])
   points(x0*1.2*max(radius)/r, y0*1.2*max(radius)/r, pch=21, bg="purple", col="purple", cex=1.1)

   pointLabel(x, y, paste(seq(1, length(x)), ":", labels=colnames(metric.matrix2)), cex=0.70, col="darkblue")
   print(cbind(colnames(metric.matrix2), x,y))

   }
      
   dev.off()

   time2 <- proc.time()
   print(paste("Total time:", signif(sum((time2 - time1)[1:2]), digits=3), " secs"))

 }

#-------------------------------------------------------------------------------------------------
   CCBA_assoc.metric.v1 <- function(
   #
   # Find association between a target and feature
   # P. Tamayo Jan 17, 2016
   #
       target,
       feature,
       cond.feature  = NULL,
       type          = "IC",
       knn           = 3,
       n.grid        = 25)
   {

     locs <- seq(1, length(feature))[!is.na(feature)]
     if (length(locs) <= 1) return(0)
     feature <- feature[locs]
     target <- target[locs]
     
      if (type=="IC") { # Information Coefficient (Kernel method) 
        return(CCBA_IC.v1(target, feature, n.grid=n.grid))
      } else if (type=="ICR") { # Information Coefficient (Kernel method) of "Ranked" profiles
        target2 <- rank(target)
        feature2 <- rank(feature)           
        return(CCBA_IC.v1(target2, feature2, n.grid=n.grid))
      } else if (type=="DIFF.MEANS") {
         x <- split(feature, target)
         m1 <- mean(x[[order(names(x), decreasing=T)[1]]])
         m2 <- mean(x[[order(names(x), decreasing=T)[2]]])
         return(m1 - m2)
      } else if (type=="AUC.ROC") {
         target <- match(target, sort(unique(target))) - 1
         perf.auc <- roc.area(obs=target, pred= (feature - min(feature))/(max(feature) - min(feature)))
         return(perf.auc$A)
      } else if (type=="AUC.REC") {
         perf.auc <- CCBA_rec_area(obs=target, pred= (feature - min(feature))/(max(feature) - min(feature)), metric = "squared.error")
         return(perf.auc$A)
      } else if (type=="IC.DM") {  # IC times the difference of medians (this is useful for signatures from isogenic samples)
         IC <- CCBA_IC.v1(target, feature, n.grid=n.grid)
         x <- split(feature, target)
         m1 <- median(x[[order(names(x), decreasing=T)[1]]])
         m2 <- median(x[[order(names(x), decreasing=T)[2]]])
         return(abs(IC)*(m1 - m2))
      } else if (type=="DIFF.MEDIANS") {
         x <- split(feature, target)
         m1 <- median(x[[order(names(x), decreasing=T)[1]]])
         m2 <- median(x[[order(names(x), decreasing=T)[2]]])
         return(m1 - m2)
      } else if (type=="T.TEST") {
         x <- split(feature, target)
         return(t.test(x=x[[order(names(x), decreasing=T)[1]]], y=x[[order(names(x), decreasing=T)[2]]])$statistic)
      } else if (type=="IC.S2N.M") {
         IC <- CCBA_IC.v1(target, feature, n.grid=n.grid)
         x <- split(feature, target)
         m1 <- median(x[[order(names(x), decreasing=T)[1]]])
         m2 <- median(x[[order(names(x), decreasing=T)[2]]])
         s1 <- ifelse(length(x[[order(names(x), decreasing=T)[1]]]) > 1, mad(x[[order(names(x), decreasing=T)[1]]]), 0)
         s2 <- ifelse(length(x[[order(names(x), decreasing=T)[2]]]) > 1, mad(x[[order(names(x), decreasing=T)[2]]]), 0)
         s1 <- ifelse(s1 < 0.1*abs(m1), 0.1*abs(m1), s1)
         s2 <- ifelse(s2 < 0.1*abs(m2), 0.1*abs(m2), s2)
         return(abs(IC) * (m1 - m2)/(s1 + s2 + 0.1))
      } else if (type=="S2N") {
         x <- split(feature, target)
         m1 <- mean(x[[order(names(x), decreasing=T)[1]]])
         m2 <- mean(x[[order(names(x), decreasing=T)[2]]])
         s1 <- ifelse(length(x[[order(names(x), decreasing=T)[1]]]) > 1, sd(x[[order(names(x), decreasing=T)[1]]]), 0)
         s2 <- ifelse(length(x[[order(names(x), decreasing=T)[2]]]) > 1, sd(x[[order(names(x), decreasing=T)[2]]]), 0)
         s1 <- ifelse(s1 < 0.1*abs(m1), 0.1*abs(m1), s1)
         s2 <- ifelse(s2 < 0.1*abs(m2), 0.1*abs(m2), s2)
         return((m1 - m2)/(s1 + s2 + 0.1))
      } else if (type=="CORR") {           
         return(cor(target, feature))
      } else if (type=="SPEAR") {           
         return(cor(target, feature, method = "spearman"))
      } else {
         stop(paste("Unknow assoc. metric name:", type))
      }
   }

#-------------------------------------------------------------------------------------------------
   CCBA_rec_area <- function(
   #
   # Compute area under the Regression Error Curve (REC) 
   # This is the counterpart of the ROC for continuous target values (regression) 
   # see e.g. Bi, J.; Bennett, K.P. (2003). "Regression error characteristic curves".
   # Twentieth International Conference on Machine Learning (ICML-2003). Washington, DC.
   # P. Tamayo Jan 17, 2016
       
		obs,
		pred,
		metric = "absolute.deviation",       # Either "squared.error" or "absolute.deviation"
		interval = 0.01)
   {
	error.windows = seq(0, 1, by=interval)
	n.errors = length(error.windows)
	intervals = rep( interval, n.errors )
	n.obs = length(obs)
	n.pred = length(pred)
	if( n.obs != n.pred ){ stop( "The number of observations does not equal the number of predictions." ) }
	
	if( metric == "squared.error" ){
		difference = (obs-pred)^2
		accuracy = unlist(lapply(error.windows, FUN=squared.error, difference, n.obs))
	} else if( metric == "absolute.deviation" ){
		difference = abs(obs-pred)
		accuracy = unlist(lapply(error.windows, FUN=absolute.deviation, difference, n.obs))
	}
	triangle.heights = accuracy - c(0, accuracy[1:(n.errors-1)])
	triangles = triangle.heights*intervals/2
	rectangle.heights = c(0, accuracy[1:(n.errors-1)])
	rectangles = rectangle.heights*intervals
	A = sum( rectangles + triangles)
	
	# Calculate p-value using Cramer-Von-Mises Criterion
	T2 = .25*(sum((accuracy-error.windows)^2))  # accuracy-error.windows = integral difference between null model and REC
	p.value = cvmts.pval(T2, n.errors, n.errors)

	return( list(A = A, p.value = p.value, T2=T2) )
}

#-------------------------------------------------------------------------------------------------
   CCBA_multiple_features_IC_selection.v1 <- function(
   #
   #  Match multiple genomic features (feature files) to a given target profile using the IC coefficient
   #
      input_dataset,                             # Input dataset (GCT) containing the target profile
      target,
      target.dir                  = "positive",
      directory,                                 # Directory where to produce results                                          
      identifier                  = "run1",      # string or prefix to identify this analysis
      feature.type.files,                        # List of feature typers and correponding file e.g.
                                                 # list("ACHILLES"     = "~/CGP2013/Distiller/Achilles_v2.4.1.rnai.Gs.gct",
                                                 #  "MUT_CNA"      = "~/CGP2013/Distiller/RNAseqMUTs_CNA_20130729.gct",  
                                                 #  "EXP_PATHWAYS" = "~/CGP2013/Distiller/CCLE_MSigDB_plus_oncogenic.PATHWAYS.v2.gct",
                                                 #  "EXP_GENES"    = "~/CGP2013/CCLE/rnaseq.v3.gct",
                                                 #  "RPPA"         = "~/CGP2013/Distiller/RPPA.dat.gct")
       feature.dir,                              # Direction of features matching for the feature.type.files e.g. c(0, 1, 1, 1, 1),     
       n.markers                   = 25,         # Number of top hits shown in the heatmaps
       metric                      = "IC",
       n.boot                      = 100,
       n.perm                      = 10,                                         
       locs.table.file             = NULL,
       log.table.file,
       missing.value.color         = "wheat",     # Missing feature color
       target.style                = "color.bar",          # "color.bar" or "bar.graph"       
       min.thres                   = 10,
       character.scaling           = 0.65,
       phen.table                  = NULL,
       produce.mds.plots           = T,
       phenotypes                  = NULL)        # list(list("ALL"))
     
   {
      version <- identifier
      target.file <- input_dataset
      dataset.1 <- CCBA_read_GCT_file.v1(filename = target.file)
      
      H <- data.matrix(dataset.1$ds)
      feature.types <- names(feature.type.files)
      n.f.types <- length(feature.types)
      
      if (!is.null(phen.table)) {
         samples.table <- read.delim(phen.table, header=T, row.names=1, sep="\t", skip=0)
         table.names <- row.names(samples.table)
       }
      log.table <- NULL

         target.dir.name <- paste(directory, target, sep="")
         dir.create(target.dir.name, showWarnings=FALSE)

         target.dir <- as.numeric(ifelse(target.dir == "positive", 1, 0) )
      
         for (f in 1:n.f.types) {   # loop over feature types
            n.markers.feature <- n.markers
             
            dir <- ifelse(xor(target.dir, feature.dir[[f]]), "negative", "positive")
           
            print(paste("Reading target file:", target.file))
            dataset.1 <- CCBA_read_GCT_file.v1(filename = target.file)
            sample.names.1 <- dataset.1$names
            
            print(paste("Reading features file:", feature.type.files[[f]]))
            dataset.2 <- CCBA_read_GCT_file.v1(filename = feature.type.files[[f]])
            ds2 <- data.matrix(dataset.2$ds)
            sample.names.2 <- dataset.2$names

            if (nrow(ds2) < 2*n.markers.feature) n.markers.feature <- floor(nrow(ds2)/2)

            if (is.null(phen.table)) {
               phen.set <- 1
               phen.selected <- NULL
               results.file.pdf <- paste(directory, target, "/", target, "_vs_", feature.types[[f]], "_", version, ".pdf", sep="")
               results.file.txt <- paste(directory, target, "/", target, "_vs_", feature.types[[f]], "_", version, ".txt", sep="")               
            } else {
              results.file.pdf <- paste(directory, target, "/", "_", target, "_vs_", feature.types[[f]], "_", version, ".pdf", sep="")
              results.file.txt <- paste(directory, target, "/", "_", target, "_vs_", feature.types[[f]], "_", version, ".txt", sep="")               
              if (length(phenotypes) == 1) {
                  phen.set <- unlist(phenotypes[[1]])
                  phen.selected <- "1"
               } else {
                  phen.set <- unlist(phenotypes[[d]])
                  phen.selected <- "1"
               }
             }
            for (p in 1:length(phen.set)) {  # loop over phenotypes
               if (phen.set == 1) {
                  phen <- NULL
                  print(paste("Working on:  feature=", f))
                  overlap <- intersect(sample.names.1, sample.names.2)
                  doc.string <- c(ifelse(length(overlap) < min.thres, "NO", "YES"), length(overlap), target, dir, feature.types[[f]],
                                  version, target.file, feature.type.files[[f]])
                } else {
                  phen <- phen.set[p]
                  print(paste("Working on: feature=", f, " phen=", phen))
                  sample.names.3 <- table.names[samples.table[,  phen] == "1"]
                  overlap <- intersect(sample.names.1, intersect(sample.names.2, sample.names.3))
                  doc.string <- c(ifelse(length(overlap) < min.thres, "NO", "YES"), length(overlap), phen, target, dir, feature.types[[f]],
                                  version, target.file, feature.type.files[[f]])
               }

               log.table <- rbind(log.table, doc.string)
               
               # Only run analysis if there are at least min.thres samples
         
               if (length(overlap) < min.thres) next
             
               CCBA_IC_selection.v1(
                  ds1 =                            target.file,
                  target.name =                    target,
                  ds2 =                            feature.type.files[[f]],
                  n.markers =                      n.markers.feature,             
                  n.perm =                         n.perm,           
                  permutation.test.type =          "standard",  
                  n.boot =                         n.boot,
                  seed =                           2345971, 
                  assoc.metric.type =              metric,
                  direction =                      dir,
                  sort.target =                    TRUE,           
                  results.file.pdf =               results.file.pdf,
                  results.file.txt =               results.file.txt,
                  sort.columns.inside.classes =    F,
                  locs.table.file =                locs.table.file,
                  consolidate.identical.features = "identical",
                  cons.features.hamming.thres =    0,   
                  save.matched.dataset =           F,  
                  produce.aux.histograms =         F,
                  produce.heat.map =               T,
                  produce.mds.plots =              produce.mds.plots,
                  character.scaling =              character.scaling,
                  mds.plot.type =                  "smacof",
                  n.grid =                         25,
                  target.style = target.style,          # "color.bar" or "bar.graph"                   
                  missing.value.color  =           missing.value.color,     # Missing feature color                   
                  phen.table =                     phen.table,
                  phen.column =                    phen,
                  phen.selected =                  phen.selected)
            
              } # loop over phenotypes
  
          } # loop over feature types

       # Save log records
      
       if (phen.set == 1) {
          header <- c("Run:", "Samples:", "target:", "Direction:", "Feature.type:", "Version:", "target.file:", "Feature.type.file:")
        } else {
          header <- c("Run:", "Samples:", "Phenotype:", "target:", "Direction:", "Feature.type:", "Version:", "target.file:", "Feature.type.file:")
        }
        colnames(log.table) <- header
        write.table(log.table, file=log.table.file, quote=F, col.names = T, row.names = F, append = F, sep="\t")
  }

#-------------------------------------------------------------------------------------------------       
   CCBA_write.gct.v1 <- function(
   #
   # Write data frame to a GCT file
   # P. Tamayo Jan 17, 2016
   #
       gct.data.frame,
       descs = "",
       filename) 
   {
    f <- file(filename, "w")
    cat("#1.2", "\n", file = f, append = TRUE, sep = "")
    cat(dim(gct.data.frame)[1], "\t", dim(gct.data.frame)[2], "\n", file = f, append = TRUE, sep = "")
    cat("Name", "\t", file = f, append = TRUE, sep = "")
    cat("Description", file = f, append = TRUE, sep = "")

    colnames <- colnames(gct.data.frame)
    cat("\t", colnames[1], file = f, append = TRUE, sep = "")

    if (length(colnames) > 1) {
       for (j in 2:length(colnames)) {
           cat("\t", colnames[j], file = f, append = TRUE, sep = "")
       }
     }
    cat("\n", file = f, append = TRUE, sep = "\t")

    oldWarn <- options(warn = -1)
    m <- matrix(nrow = dim(gct.data.frame)[1], ncol = dim(gct.data.frame)[2] +  2)
    m[, 1] <- row.names(gct.data.frame)
    if (length(descs) > 1) {
        m[, 2] <- descs
    } else {
        m[, 2] <- row.names(gct.data.frame)
    }
    index <- 3
    for (i in 1:dim(gct.data.frame)[2]) {
        m[, index] <- gct.data.frame[, i]
        index <- index + 1
    }
    write.table(m, file = f, append = TRUE, quote = FALSE, sep = "\t", eol = "\n", col.names = FALSE, row.names = FALSE)
    close(f)
    options(warn = 0)

}

#-------------------------------------------------------------------------------------------------       
qvalue <- function(p=NULL, lambda=seq(0,0.90,0.05), pi0.method="smoother", fdr.level=NULL, robust=FALSE, 
  gui=FALSE, smooth.df = 3, smooth.log.pi0 = FALSE) {
#Input
#=============================================================================
#p: a vector of p-values (only necessary input)
#fdr.level: a level at which to control the FDR (optional)
#lambda: the value of the tuning parameter to estimate pi0 (optional)
#pi0.method: either "smoother" or "bootstrap"; the method for automatically
#           choosing tuning parameter in the estimation of pi0, the proportion
#           of true null hypotheses
#robust: an indicator of whether it is desired to make the estimate more robust
#        for small p-values and a direct finite sample estimate of pFDR (optional)
#gui: A flag to indicate to 'qvalue' that it should communicate with the gui.  ## change by Alan
#     Should not be specified on command line.
#smooth.df: degrees of freedom to use in smoother (optional)
#smooth.log.pi0: should smoothing be done on log scale? (optional)
#
#Output
#=============================================================================
#call: gives the function call
#pi0: an estimate of the proportion of null p-values
#qvalues: a vector of the estimated q-values (the main quantity of interest)
#pvalues: a vector of the original p-values
#significant: if fdr.level is specified, an indicator of whether the q-value
#    fell below fdr.level (taking all such q-values to be significant controls
#    FDR at level fdr.level)

#Set up communication with GUI, if appropriate
#    print(sys.calls())
#    print(sys.frames())

#    if(gui) {
#        idx <- (1:sys.nframe())[as.character(sys.calls()) == "qvalue.gui()"]
#        gui.env <- sys.frames()[[idx]]
#    }

#This is just some pre-processing
    if(is.null(p))  ## change by Alan
      {qvalue.gui(); return("Launching point-and-click...")}
    if(gui & !interactive())  ## change by Alan
      gui = FALSE

    if(min(p)<0 || max(p)>1) {
      if(gui) ## change by Alan:  check for GUI
        eval(expression(postMsg(paste("ERROR: p-values not in valid range.", "\n"))), parent.frame())
      else
#        print("ERROR: p-values not in valid range.")
      return(0)
    }
    if(length(lambda)>1 && length(lambda)<4) {
      if(gui)
        eval(expression(postMsg(paste("ERROR: If length of lambda greater than 1, you need at least 4 values.",
            "\n"))), parent.frame())
      else
#        print("ERROR: If length of lambda greater than 1, you need at least 4 values.")
      return(0)
    }
    if(length(lambda)>1 && (min(lambda) < 0 || max(lambda) >= 1)) { ## change by Alan:  check for valid range for lambda
      if(gui)
        eval(expression(postMsg(paste("ERROR: Lambda must be within [0, 1).", "\n"))), parent.frame())
      else
#        print("ERROR: Lambda must be within [0, 1).")
      return(0)
    }
    m <- length(p)
#These next few functions are the various ways to estimate pi0
    if(length(lambda)==1) {
        if(lambda<0 || lambda>=1) { ## change by Alan:  check for valid range for lambda
          if(gui)
            eval(expression(postMsg(paste("ERROR: Lambda must be within [0, 1).", "\n"))), parent.frame())
          else
#            print("ERROR: Lambda must be within [0, 1).")
          return(0)
        }

        pi0 <- mean(p >= lambda)/(1-lambda)
        pi0 <- min(pi0,1)
    }
    else {
        pi0 <- rep(0,length(lambda))
        for(i in 1:length(lambda)) {
            pi0[i] <- mean(p >= lambda[i])/(1-lambda[i])
        }

        if(pi0.method=="smoother") {
            if(smooth.log.pi0)
              pi0 <- log(pi0)

            spi0 <- smooth.spline(lambda,pi0,df=smooth.df)
            pi0 <- predict(spi0,x=max(lambda))$y

            if(smooth.log.pi0)
              pi0 <- exp(pi0)
            pi0 <- min(pi0,1)
        }
        else if(pi0.method=="bootstrap") {
            minpi0 <- min(pi0)
            mse <- rep(0,length(lambda))
            pi0.boot <- rep(0,length(lambda))
            for(i in 1:100) {
                p.boot <- sample(p,size=m,replace=TRUE)
                for(i in 1:length(lambda)) {
                    pi0.boot[i] <- mean(p.boot>lambda[i])/(1-lambda[i])
                }
                mse <- mse + (pi0.boot-minpi0)^2
            }
            pi0 <- min(pi0[mse==min(mse)])
            pi0 <- min(pi0,1)
        }
        else {  ## change by Alan: check for valid choice of 'pi0.method' (only necessary on command line)
#            print("ERROR: 'pi0.method' must be one of 'smoother' or 'bootstrap'.")
            return(0)
        }
    }
    if(pi0 <= 0) {
      if(gui)
        eval(expression(postMsg(
            paste("ERROR: The estimated pi0 <= 0. Check that you have valid p-values or use another lambda method.",
                "\n"))), parent.frame())
      else
#        print("ERROR: The estimated pi0 <= 0. Check that you have valid p-values or use another lambda method.")
      return(0)
    }
    if(!is.null(fdr.level) && (fdr.level<=0 || fdr.level>1)) {  ## change by Alan:  check for valid fdr.level
      if(gui)
        eval(expression(postMsg(paste("ERROR: 'fdr.level' must be within (0, 1].", "\n"))), parent.frame())
      else
#        print("ERROR: 'fdr.level' must be within (0, 1].")
      return(0)
    }
#The estimated q-values calculated here
    u <- order(p)

    # change by Alan
    # ranking function which returns number of observations less than or equal
    qvalue.rank <- function(x) {
      idx <- sort.list(x)

      fc <- factor(x)
      nl <- length(levels(fc))
      bin <- as.integer(fc)
      tbl <- tabulate(bin)
      cs <- cumsum(tbl)
 
      tbl <- rep(cs, tbl)
      tbl[idx] <- tbl

      return(tbl)
    }

    v <- qvalue.rank(p)
    
    qvalue <- pi0*m*p/v
    if(robust) {
        qvalue <- pi0*m*p/(v*(1-(1-p)^m))
    }
    qvalue[u[m]] <- min(qvalue[u[m]],1)
    for(i in (m-1):1) {
    qvalue[u[i]] <- min(qvalue[u[i]],qvalue[u[i+1]],1)
    }
#The results are returned
    if(!is.null(fdr.level)) {
        retval <- list(call=match.call(), pi0=pi0, qvalues=qvalue, pvalues=p, fdr.level=fdr.level, ## change by Alan
          significant=(qvalue <= fdr.level), lambda=lambda)
    }
    else {
        retval <- list(call=match.call(), pi0=pi0, qvalues=qvalue, pvalues=p, lambda=lambda)
    }
    class(retval) <- "qvalue"
    return(retval)
}

  SE_assoc <- function(x, y) {
           return(2 - sqrt(mean((x - y)^2)))
       }

#-------------------------------------------------------------------------------------------------       
   CCBA_REVEALER.v1 <- function(
   # REVEALER (Repeated Evaluation of VariablEs conditionAL Entropy and Redundancy) is an analysis method specifically suited
   # to find groups of genomic alterations that match in a complementary way, a predefined functional activation, dependency of
   # drug response “target” profile. The method starts by considering, if available, already known genomic alterations (“seed”)
   # that are the known “causes” or are known or assumed “associated” with the target. REVEALER starts by finding the genomic
   # alteration that best matches the target profile “conditional” to the known seed profile using the conditional mutual information.
   # The newly discovered alteration is then merged with the seed to form a new “summary” feature, and then the process repeats itself
   # finding additional complementary alterations that explain more and more of the target profile.

   ds1,                                          # Dataset that contains the "target"
   target.name,                                  # Target feature (row in ds1)
   target.match = "positive",                    # Use "positive" to match the higher values of the target, "negative" to match the lower values
   ds2,                                          # Features dataset.
   seed.names = NULL,                            # Seed(s) name(s)
   exclude.features = NULL,                      # Features to exclude for search iterations
   max.n.iter = 5,                               # Maximun number of iterations
   pdf.output.file,                              # PDF output file
   rank.target        = F,            # Use ranks of target values instead of target itself       
   count.thres.low = NULL,                       # Filter out features with less than count.thres.low 1's
   count.thres.high = NULL,                      # Filter out features with more than count.thres.low 1's

   n.markers = 30,                               # Number of top hits to show in heatmap for every iteration
   locs.table.file = NULL)                       # Table with chromosomal location for each gene symbol (optional)

{  # Additional internal settings
    
   identifier = "REVEALER"                      # Documentation suffix to be added to output file    
   n.perm = 10                                  # Number of permutations (x number of genes) for computing p-vals and FRDs
   save_preprocessed_features_dataset = NULL    # Save preprocessed features dataset    
   seed.combination.op = "max"                  # Operation to consolidate and summarize seeds to one vector of values
   assoc.metric = "IC"                          # Assoc. Metric: "IC" information coeff.; "COR" correlation.
   normalize.features = F                       # Feature row normalization: F or "standardize" or "0.1.rescaling"
   top.n = 1                                    # Number of top hits in each iteration to diplay in Landscape plot
   max.n = 2                                    # Maximum number of iterations to diplay in Landscape plot
   phen.table = NULL                            # Table with phenotypes for each sample (optional)
   phen.column = NULL                           # Column in phen.table containing the relevant phenotype info
   phen.selected = NULL                         # Use only samples of these phenotypes in REVEALER analysis
   produce.lanscape.plot = F                    # Produce multi-dimensional scaling projection plot
   character.scaling = 1.25                     # Character scaling for heatmap
   r.seed = 34578                               # Random number generation seed
   consolidate.identical.features = F           # Consolidate identical features: F or "identical" or "similar" 
   cons.features.hamming.thres = NULL           # If consolidate.identical.features = "similar" then consolidate features within this Hamming dist. thres.

# ------------------------------------------------------------------------------------------------------------------------------
   print(paste("ds1:", ds1))
   print(paste("target.name:", target.name))
   print(paste("target.match:", target.match))
   print(paste("ds2:", ds2))                        
   print(paste("seed.names:", seed.names))
#   print(paste("exclude.features:", exclude.features))
   print(paste("max.n.iter:", max.n.iter))
   print(paste("pdf.output.file:", pdf.output.file))
#   print(paste("n.perm:", n.perm))                  
   print(paste("count.thres.low:", count.thres.low))
   print(paste("count.thres.high:", count.thres.high))
#   print(paste("identifier:", identifier))                  
   print(paste("n.markers:", n.markers))   
   print(paste("locs.table.file:", locs.table.file))

# ------------------------------------------------------------------------------------------------------------------------------
   # Load libraries

   if (is.null(seed.names)) seed.names <- "NULLSEED"
   
   pdf(file=pdf.output.file, height=14, width=8.5)
   set.seed(r.seed)

   # Read table with HUGO gene symbol vs. chr location
   
   if (!is.null(locs.table.file)) {
      locs.table <- read.table(locs.table.file, header=T, sep="\t", skip=0, colClasses = "character")
    }
   
   # Define color map

   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)

   # Read datasets

   dataset.1 <- CCBA_read_GCT_file.v1(filename = ds1)
   m.1 <- data.matrix(dataset.1$ds)
   row.names(m.1) <- dataset.1$row.names
   Ns.1 <- ncol(m.1)  
   sample.names.1 <- colnames(m.1) <- dataset.1$names

   dataset.2 <- CCBA_read_GCT_file.v1(filename = ds2)
   m.2 <- data.matrix(dataset.2$ds)
   row.names(m.2) <- dataset.2$row.names
   Ns.2 <- ncol(m.2)  
   sample.names.2 <- colnames(m.2) <- dataset.2$names

    # exclude samples with target == NA

   target <- m.1[target.name,]
   print(paste("initial target length:", length(target)))      
   locs <- seq(1, length(target))[!is.na(target)]
   m.1 <- m.1[,locs]
   sample.names.1 <- sample.names.1[locs]
   print(paste("target length after excluding NAs:", ncol(m.1)))     

   overlap <- intersect(sample.names.1, sample.names.2)
   length(overlap)
   locs1 <- match(overlap, sample.names.1)
   locs2 <- match(overlap, sample.names.2)
   m.1 <- m.1[, locs1]
   m.2 <- m.2[, locs2]

   # Filter samples with only the selected phenotypes 

   if (!is.null(phen.selected)) {
      samples.table <- read.table(phen.table, header=T, row.names=1, sep="\t", skip=0)
      table.sample.names <- row.names(samples.table)
      locs1 <- match(colnames(m.2), table.sample.names)
      phenotype <- as.character(samples.table[locs1, phen.column])
      
      locs2 <- NULL
      for (k in 1:ncol(m.2)) {   
         if (!is.na(match(phenotype[k], phen.selected))) {
            locs2 <- c(locs2, k)
         }
      }
      length(locs2)
      m.1 <- m.1[, locs2]
      m.2 <- m.2[, locs2]
      phenotype <- phenotype[locs2]
      table(phenotype)
    }

   # Define target

   target <- m.1[target.name,]
   if(rank.target == T) target <- rank(target)   
   if (target.match == "negative") {
      ind <- order(target, decreasing=F)
   } else {
      ind <- order(target, decreasing=T)
   }
   target <- target[ind]
   m.2 <- m.2[, ind]

   if (!is.na(match(target.name, row.names(m.2)))) {
     loc <- match(target.name, row.names(m.2))
     m.2 <- m.2[-loc,]
   }

   MUT.count <- AMP.count <- DEL.count <- 0
   for (i in 1:nrow(m.2)) {
      temp <- strsplit(row.names(m.2)[i], split="_")
      temp <- strsplit(temp[[1]][length(temp[[1]])], split=" ")
      suffix <- temp[[1]][1]
      if (!is.na(suffix)) {
         if (suffix == "MUT") MUT.count <- MUT.count + 1
         if (suffix == "AMP") AMP.count <- AMP.count + 1
         if (suffix == "DEL") DEL.count <- DEL.count + 1
     }
   }
   print(paste("Initial number of features ", nrow(m.2), " MUT:", MUT.count, " AMP:", AMP.count, " DEL:", DEL.count))
   
   # Eliminate flat, sparse or features that are too dense

   if (!is.null(count.thres.low) && !is.null(count.thres.high)) {
      sum.rows <- rowSums(m.2)
      seed.flag <- rep(0, nrow(m.2))
      if (seed.names != "NULLSEED") {
         locs <- match(seed.names, row.names(m.2))
         locs <- locs[!is.na(locs)]
         seed.flag[locs] <- 1
      }
      retain <- rep(0, nrow(m.2))
      for (i in 1:nrow(m.2)) {
         if ((sum.rows[i] >= count.thres.low) && (sum.rows[i] <= count.thres.high)) retain[i] <- 1
         if (seed.flag[i] == 1) retain[i] <- 1
      }

      m.2 <- m.2[retain == 1,]
      print(paste("Number of features kept:", sum(retain), "(", signif(100*sum(retain)/length(retain), 3), " percent)"))
  }
   
   # Normalize features and define seeds

  if (normalize.features == "standardized") {
      for (i in 1:nrow(m.2)) {
         mean.row <- mean(m.2[i,])
         sd.row <- ifelse(sd(m.2[i,]) == 0, 0.1*mean.row, sd(m.2[i,]))
         m.2[i,] <- (m.2[i,] - mean.row)/sd.row
       }
   } else if (normalize.features == "0.1.rescaling") {
      for (i in 1:nrow(m.2)) {
         max.row <- max(m.2[i,])
         min.row <- min(m.2[i,])
         range.row <- ifelse(max.row == min.row, 1, max.row - min.row)
         m.2[i,] <- (m.2[i,] - min.row)/range.row
       }
    }
   
  if (seed.names == "NULLSEED") {
     seed <- as.vector(rep(0, ncol(m.2)))      
     seed.vectors <- as.matrix(t(seed))
  } else {
      print("Location(s) of seed(s):")
      print(match(seed.names, row.names(m.2)))
      if (length(seed.names) > 1) {
         seed <- apply(m.2[seed.names,], MARGIN=2, FUN=seed.combination.op)
         seed.vectors <- as.matrix(m.2[seed.names,])
      } else {
         seed <- m.2[seed.names,]
         seed.vectors <- as.matrix(t(m.2[seed.names,]))
      }
      locs <- match(seed.names, row.names(m.2))
      locs
     m.2 <- m.2[-locs,]
     dim(m.2)
   }

  if (length(table(m.2[1,])) > ncol(m.2)*0.5) { # continuous target
     feature.type <- "continuous"
  } else {
     feature.type <- "discrete"
  }
    
  # Exclude user-specified features 
   
   if (!is.null(exclude.features)) {
      locs <- match(exclude.features, row.names(m.2))
      locs <- locs[!is.na(locs)]
      m.2 <- m.2[-locs,]
    }

  #  Consolidate identical features

  # This is a very fast way to eliminate perfectly identical features compared with what we do below in "similar"
   
   if (consolidate.identical.features == "identical") {  
      dim(m.2)
      summary.vectors <- apply(m.2, MARGIN=1, FUN=paste, collapse="")
      ind <- order(summary.vectors)
      summary.vectors <- summary.vectors[ind]
      m.2 <- m.2[ind,]
      taken <- i.count <- rep(0, length(summary.vectors))
      i <- 1
      while (i <= length(summary.vectors)) {
        j <- i + 1
        while ((summary.vectors[i] == summary.vectors[j]) & (j <= length(summary.vectors))) {
            j <- j + 1
         }
        i.count[i] <- j - i
        if (i.count[i] > 1) taken[seq(i + 1, j - 1)] <- 1
        i <- j
      }
   
      if (sum(i.count) != length(summary.vectors)) stop("ERROR")     # Add counts in parenthesis
      row.names(m.2) <- paste(row.names(m.2), " (", i.count, ")", sep="")
      m.2 <- m.2[taken == 0,]
      dim(m.2)

   # This uses the hamming distance to consolidate similar features up to the Hamming dist. threshold 
      
   } else if (consolidate.identical.features == "similar") { 
      hamming.matrix <- hamming.distance(m.2)
      taken <- rep(0, nrow(m.2))
      for (i in 1:nrow(m.2)) {
         if (taken[i] == 0) { 
            similar.features <- row.names(m.2)[hamming.matrix[i,] <= cons.features.hamming.thres]
            if (length(similar.features) > 1) {
               row.names(m.2)[i]  <- paste(row.names(m.2)[i], " [", length(similar.features), "]", sep="") # Add counts in brackets
               locs <- match(similar.features, row.names(m.2))
               taken[locs] <- 1
               taken[i] <- 0
            }
        }
      }
      m.2 <- m.2[taken == 0,]
     dim(m.2)
   }

   MUT.count <- AMP.count <- DEL.count <- 0
   for (i in 1:nrow(m.2)) {
      temp <- strsplit(row.names(m.2)[i], split="_")
      temp <- strsplit(temp[[1]][length(temp[[1]])], split=" ")      
      suffix <- temp[[1]][1]
      if (!is.na(suffix)) {
         if (suffix == "MUT") MUT.count <- MUT.count + 1
         if (suffix == "AMP") AMP.count <- AMP.count + 1
         if (suffix == "DEL") DEL.count <- DEL.count + 1
     }
   }
   print(paste("Number of features (after filtering and consolidation)",
   nrow(m.2), " MUT:", MUT.count, " AMP:", AMP.count, " DEL:", DEL.count))
   
   # Add location info

   if (!is.null(locs.table.file)) {
      gene.symbol <- row.names(m.2)
      chr <- rep(" ", length(gene.symbol))
      for (i in 1:length(gene.symbol)) {
        temp1 <- strsplit(gene.symbol[i], split="_")
        temp2 <- strsplit(temp1[[1]][1], split="\\.")
        gene.symbol[i] <- ifelse(temp2[[1]][1] == "", temp1[[1]][1], temp2[[1]][1])
        loc <- match(gene.symbol[i], locs.table[,"Approved.Symbol"])
        chr[i] <- ifelse(!is.na(loc), locs.table[loc, "Chromosome"], " ")
       }
      row.names(m.2)  <- paste(row.names(m.2), " ", chr, " ", sep="")
      print(paste("Total unmatched to chromosomal locations:", sum(chr == " "), "out of ", nrow(m.2), "features"))
    }

   # Save filtered and consolidated file

    if (!is.null(save_preprocessed_features_dataset)) {
       CCBA_write.gct.v1(gct.data.frame = data.frame(m.2), descs = row.names(m.2), filename = save_preprocessed_features_dataset)
   }
   
   # Compute MI and % explained with original seed(s)
   
   median_target <- median(target)
    if (target.match == "negative") {
      target.locs <- seq(1, length(target))[target <= median_target]
    } else {
      target.locs <- seq(1, length(target))[target > median_target]
    }

   cmi.orig.seed <- cmi.orig.seed.cum <- pct_explained.orig.seed <- pct_explained.orig.seed.cum <- vector(length=length(seed.names), mode="numeric")
   if (length(seed.names) > 1) {
      seed.cum <- NULL
      for (i in 1:nrow(seed.vectors)) {
         y <- seed.vectors[i,]
         cmi.orig.seed[i] <- CCBA_assoc.v1(target, y, assoc.metric)
         pct_explained.orig.seed[i] <- sum(y[target.locs])/length(target.locs)
         seed.cum <- apply(rbind(seed.vectors[i,], seed.cum), MARGIN=2, FUN=seed.combination.op)
         cmi.orig.seed.cum[i] <- CCBA_assoc.v1(target, seed.cum, assoc.metric)
         pct_explained.orig.seed.cum[i] <- sum(seed.cum[target.locs])/length(target.locs)
      }
   } else {
       y <- as.vector(seed.vectors)
       seed.cum <- y
       cmi.orig.seed <- cmi.orig.seed.cum <- CCBA_assoc.v1(target, y, assoc.metric)
       pct_explained.orig.seed <- sum(y[target.locs])/length(target.locs)
   }
    cmi.seed.iter0 <- CCBA_assoc.v1(target, seed, assoc.metric)
    pct_explained.seed.iter0 <- sum(seed[target.locs])/length(target.locs) 

   # CMI iterations

   cmi <- pct_explained <- cmi.names <- matrix(0, nrow=nrow(m.2), ncol=max.n.iter)
   cmi.seed <- pct_explained.seed <- vector(length=max.n.iter, mode="numeric")
   seed.names.iter <- vector(length=max.n.iter, mode="character")
   seed.initial <- seed
   seed.iter <- matrix(0, nrow=max.n.iter, ncol=ncol(m.2))

   target.rand <- matrix(target, nrow=n.perm, ncol=ncol(m.2), byrow=TRUE)
   for (i in 1:n.perm) target.rand[i,] <- sample(target.rand[i,])

   for (iter in 1:max.n.iter) {

      cmi.rand <- matrix(0, nrow=nrow(m.2), ncol=n.perm)     
      for (k in 1:nrow(m.2)) {
         if (k %% 100 == 0) print(paste("Iter:", iter, " feature #", k, " out of ", nrow(m.2)))
         y <- m.2[k,]
         cmi[k, iter] <- CCBA_cond.assoc.v1(target, y, seed, assoc.metric)
         for (j in 1:n.perm) {
            cmi.rand[k, j] <- CCBA_cond.assoc.v1(target.rand[j,], y, seed, assoc.metric)
          }
       }

      if (target.match == "negative") {
         ind <- order(cmi[, iter], decreasing=F)
      } else {
         ind <- order(cmi[, iter], decreasing=T)
      }
      cmi[, iter] <- cmi[ind, iter]
      cmi.names[, iter] <- row.names(m.2)[ind]
      pct_explained[iter] <- sum(m.2[cmi.names[1, iter], target.locs])/length(target.locs)
      
      # Estimate p-vals and FDRs

      p.val <- FDR <- FDR.lower <- rep(0, nrow(m.2))
      for (i in 1:nrow(m.2)) p.val[i] <- sum(cmi.rand >  cmi[i, iter])/(nrow(m.2)*n.perm)
      FDR <- p.adjust(p.val, method = "fdr", n = length(p.val))
      FDR.lower <- p.adjust(1 - p.val, method = "fdr", n = length(p.val))
    
      for (i in 1:nrow(m.2)) {
         if (cmi[i, iter] < 0) {
            p.val[i] <- 1 - p.val[i]
            FDR[i] <- FDR.lower[i]
         }
         p.val[i] <- signif(p.val[i], 2)
         FDR[i] <- signif(FDR[i], 2)
      }
      p.zero.val <- paste("<", signif(1/(nrow(m.2)*n.perm), 2), sep="")
      p.val <- ifelse(p.val == 0, rep(p.zero.val, length(p.val)), p.val)

      # Make a heatmap of the n.marker top hits in this iteration

      size.mid.panel <- length(seed.names) + iter
      pad.space <- 15
      
      nf <- layout(matrix(c(1, 2, 3, 4), 4, 1, byrow=T), 1, c(2, size.mid.panel, ceiling(n.markers/2) + 4, pad.space), FALSE)
      cutoff <- 2.5
      x <- as.numeric(target)         
      x <- (x - mean(x))/sd(x)         
      ind1 <- which(x > cutoff)
      ind2 <- which(x < -cutoff)
      x[ind1] <- cutoff
      x[ind2] <- -cutoff
      V1 <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
      par(mar = c(1, 22, 2, 12))
      image(1:length(target), 1:1, as.matrix(V1), col=mycol, zlim=c(0, ncolors), axes=FALSE,
            main=paste("REVEALER - Iteration:", iter), sub = "", xlab= "", ylab="",
            font=2, family="")
      axis(2, at=1:1, labels=paste("TARGET: ", target.name), adj= 0.5, tick=FALSE,las = 1, cex=1,
           cex.axis=character.scaling, font.axis=1,
           line=0, font=2, family="")
      axis(4, at=1:1, labels="  IC ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="black")
      axis(4, at=1:1, labels="       / CIC", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="steelblue")
 
      if (iter == 1) {
            V0 <- rbind(seed.vectors, seed + 2)
            cmi.vals <- c(cmi.orig.seed, cmi.seed.iter0)
            cmi.vals <- signif(cmi.vals, 2)
            cmi.cols <- rep("black", length(cmi.orig.seed) + 1)                     # IC/CIC colors
            row.names(V0) <- c(paste("SEED: ", seed.names), "SUMMARY SEED:")
            V0 <- apply(V0, MARGIN=2, FUN=rev)
       } else {
         V0 <- rbind(seed.vectors, m.2[seed.names.iter[1:(iter-1)],], seed + 2)
         row.names(V0) <- c(paste("SEED:   ", seed.names), paste("ITERATION ", seq(1, iter-1), ":  ",
                                                                 seed.names.iter[1:(iter-1)], sep=""), "SUMMARY SEED:")
         cmi.vals <- c(cmi.orig.seed, cmi[1, 1:iter-1], cmi.seed[iter-1])
         cmi.vals <- signif(cmi.vals, 2)
         cmi.cols <- c(rep("black", length(cmi.orig.seed)), rep("steelblue", iter - 1), "black")      # IC/CIC colors
         pct.vals <- c(signif(pct_explained.orig.seed, 2), signif(pct_explained[seq(1, iter - 1)], 2),
                       signif(pct_explained.seed[iter - 1], 2))         
         V0 <- apply(V0, MARGIN=2, FUN=rev)
       }

      all.vals <- cmi.vals
      par(mar = c(1, 22, 0, 12))
      if (feature.type == "discrete") {
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 3), col=c(brewer.pal(9, "Blues")[3], brewer.pal(9, "Blues")[9],
                                                                    brewer.pal(9, "Greys")[2], brewer.pal(9, "Greys")[5]),
               axes=FALSE, main="", sub = "", xlab= "", ylab="")
      } else { # continuous
         for (i in 1:length(V0[,1])) {
            x <- as.numeric(V0[i,])
            V0[i,] <- (x - mean(x))/sd(x)
            max.v <- max(max(V0[i,]), -min(V0[i,]))
            V0[i,] <- ceiling(ncolors * (V0[i,] - (- max.v))/(1.001*(max.v - (- max.v))))
         }
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE, main="", sub = "",
               xlab= "", ylab="")
      }
#      axis(2, at=1:nrow(V0), labels=row.names(V0), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#           line=0, font=2, family="")
#      axis(4, at=1:nrow(V0), labels=rev(all.vals), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#           line=0, font=2, family="")
      for (axis.i in 1:nrow(V0)) {
          axis(2, at=axis.i, labels=row.names(V0)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
          axis(4, at=axis.i, labels=rev(all.vals)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
      }
      V0 <- m.2[cmi.names[1:n.markers, iter],]
      V0 <- apply(V0, MARGIN=2, FUN=rev)
      par(mar = c(6, 22, 3, 12))
      if (feature.type == "discrete") {
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 1),
               col=c(brewer.pal(9, "Blues")[3], brewer.pal(9, "Blues")[9]),
               axes=FALSE, main=paste("Top", n.markers, "Matches"), sub = "", xlab= "", ylab="")
      } else { # continuous
         for (i in 1:length(V0[,1])) {
            cutoff <- 2.5
            x <- as.numeric(V0[i,])            
            x <- (x - mean(x))/sd(x)         
            ind1 <- which(x > cutoff)
            ind2 <- which(x < -cutoff)
            x[ind1] <- cutoff
            x[ind2] <- -cutoff
            V0[i,] <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
         }
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE, main="", sub = "",
               xlab= "", ylab="")
      }
      axis(2, at=1:nrow(V0), labels=row.names(V0), adj= 0.5, tick=FALSE, las = 1, cex.axis=0.9*character.scaling,
           line=0, font=2, family="")

      all.vals <- paste(signif(cmi[1:n.markers, iter], 2), p.val[1:n.markers], FDR[1:n.markers], sep="   ")

      axis(4, at=nrow(V0)+0.4, labels=" CIC    p-val   FDR", adj= 0.5, tick=FALSE, las = 1,
           cex.axis=0.8*character.scaling, line=0, font=2, family="")
      axis(4, at=c(seq(1, nrow(V0) - 1), nrow(V0) - 0.2), labels=rev(all.vals), adj= 0.5, tick=FALSE,
           las = 1, cex.axis=0.9*character.scaling, line=0, font=2, family="")
      axis(1, at=1:ncol(V0), labels=colnames(V0), adj= 0.5, tick=FALSE,las = 3, cex=1,
           cex.axis=0.45*character.scaling,  line=0, font=2, family="")

     # second page shows the same markers clustered in groups with similar profiles

     tab <- m.2[cmi.names[1:n.markers, iter],]
     all.vals <- paste(signif(cmi[1:n.markers, iter], 2), p.val[1:n.markers], FDR[1:n.markers], sep="   ")

     # Cluster and make heatmap of n.markers top hits in groups

     tab2 <- tab + 0.001

     k.min <- 2
     k.max <- 10
     NMF.models <- nmf(tab2, seq(k.min, k.max), nrun = 25, method="brunet", seed=9876)
     plot(NMF.models)
     NMF.sum <- summary(NMF.models)

     k.vec <- seq(k.min, k.max, 1)
     cophen <- NMF.sum[, "cophenetic"]

     peak <- c(0, rep(0, k.max-2), 0)
     for (h in 2:(length(cophen) - 1)) if (cophen[h - 1] < cophen[h] & cophen[h] > cophen[h + 1]) peak[h] <- 1

     if (sum(peak) == 0) {
        if (cophen[1] > cophen[length(cophen)]) {
           k <- k.min
         } else {
           k <- k.max
         }
     } else {
        k.peaks <- k.vec[peak == 1]
        k <- rev(k.peaks)[1]
     }
     print(paste("Number of groups:", k))
     NMF.model <- nmf(tab2, k, method="brunet", seed=9876)
     classes <- predict(NMF.model, "rows")
     table(classes)
     lens <- table(classes)

     lens2 <- ifelse(lens <= 5, 5, lens)
     lens2[length(lens2)] <- lens2[length(lens2)] + 5


     def.par <- par(no.readonly = TRUE)       
     nf <- layout(matrix(seq(1, k+3), k+3, 1, byrow=T), 1, c(3.5, size.mid.panel, lens2, pad.space), FALSE)      

      cutoff <- 2.5
      x <- as.numeric(target)         
      x <- (x - mean(x))/sd(x)         
      ind1 <- which(x > cutoff)
      ind2 <- which(x < -cutoff)
      x[ind1] <- cutoff
      x[ind2] <- -cutoff
      V1 <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
      par(mar = c(1, 22, 1, 12))
      image(1:length(target), 1:1, as.matrix(V1), col=mycol, zlim=c(0, ncolors), axes=FALSE,
            main=paste("REVEALER - Iteration:", iter), sub = "", xlab= "", ylab="",
            font=2, family="")
     axis(2, at=1:1, labels=paste("TARGET: ", target.name), adj= 0.5, tick=FALSE,las = 1, cex=1,
          cex.axis=character.scaling, font.axis=1,
           line=0, font=2, family="")

      axis(4, at=1:1, labels="  IC ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="black")
      axis(4, at=1:1, labels="       / CIC", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="steelblue")

#      axis(4, at=1:1, labels="  IC/CIC", adj= 0.5, tick=FALSE, las = 1, cex.axis=0.8*character.scaling,
#           font.axis=1, line=0, font=2, family="")
 
      if (iter == 1) {
            V0 <- rbind(seed.vectors, seed + 2)
            cmi.vals <- c(cmi.orig.seed, cmi.seed.iter0)
            cmi.vals <- signif(cmi.vals, 2)
            row.names(V0) <- c(paste("SEED: ", seed.names), "SUMMARY SEED:")
            V0 <- apply(V0, MARGIN=2, FUN=rev)
       } else {
         V0 <- rbind(seed.vectors, m.2[seed.names.iter[1:(iter-1)],], seed + 2)
         row.names(V0) <- c(paste("SEED:   ", seed.names), paste("ITERATION ", seq(1, iter-1), ":  ",
                                        seed.names.iter[1:(iter-1)], sep=""), "SUMMARY SEED:")
         cmi.vals <- c(cmi.orig.seed, cmi[1, 1:iter-1], cmi.seed[iter-1])
         cmi.vals <- signif(cmi.vals, 2)
         pct.vals <- c(signif(pct_explained.orig.seed, 2), signif(pct_explained[seq(1, iter - 1)], 2),
                       signif(pct_explained.seed[iter - 1], 2))         
         V0 <- apply(V0, MARGIN=2, FUN=rev)
       }

      all.vals <- cmi.vals
      par(mar = c(1, 22, 0, 12))
      if (feature.type == "discrete") {
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 3), col=c(brewer.pal(9, "Blues")[3], brewer.pal(9, "Blues")[9],
                                                                    brewer.pal(9, "Greys")[2], brewer.pal(9, "Greys")[5]),
               axes=FALSE, main="", sub = "", xlab= "", ylab="")
      } else { # continuous
         for (i in 1:length(V0[,1])) {
            x <- as.numeric(V0[i,])
            V0[i,] <- (x - mean(x))/sd(x)
            max.v <- max(max(V0[i,]), -min(V0[i,]))
            V0[i,] <- ceiling(ncolors * (V0[i,] - (- max.v))/(1.001*(max.v - (- max.v))))
         }
         image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE, main="",
               sub = "", xlab= "", ylab="")
      }
      
#      axis(2, at=1:nrow(V0), labels=row.names(V0), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#           line=0, font=2, family="")
#      axis(4, at=1:nrow(V0), labels=rev(all.vals), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#           line=0, font=2, family="")
      for (axis.i in 1:nrow(V0)) {
          axis(2, at=axis.i, labels=row.names(V0)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
          axis(4, at=axis.i, labels=rev(all.vals)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
      }

    # Groups of abnormalities
      
     all.vals <- paste(signif(cmi[1:n.markers, iter], 2), p.val[1:n.markers], FDR[1:n.markers], sep="   ")
      
     for (h in sort(unique(classes))) {      
         if (lens[h] == 1) {
            V0 <- t(as.matrix(tab[classes == h,]))
          } else {
            V0 <- tab[classes == h,]
            V0 <- apply(V0, MARGIN=2, FUN=rev)
          }
         r.names <- row.names(tab)[classes == h]
         all.vals0 <- all.vals[classes == h]         
         if (h < k) {
            par(mar = c(0.5, 22, 1, 12))
          } else {
            par(mar = c(3, 22, 1, 12))
          }
         if (feature.type == "discrete") {
            if (lens[h] == 1) {           
               image(1:ncol(V0), 1, t(V0), zlim = c(0, 1), col=c(brewer.pal(9, "Blues")[3],
                                                               brewer.pal(9, "Blues")[9]),
                     axes=FALSE, main=paste("Top Matches. Group:", h, "(iter ", iter, ")"), sub = "",
                     xlab= "", ylab="", cex.main=0.8)
             } else {
               image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 1), col=c(brewer.pal(9, "Blues")[3],
                                                                        brewer.pal(9, "Blues")[9]),
                     axes=FALSE, main=paste("Top Matches. Group:", h, "(iter ", iter, ")"),
                     sub = "", xlab= "", ylab="", cex.main=0.8)
             }
         } else { # continuous
            for (i in 1:length(V0[,1])) {
               cutoff <- 2.5
               x <- as.numeric(V0[i,])            
               x <- (x - mean(x))/sd(x)         
               ind1 <- which(x > cutoff)
               ind2 <- which(x < -cutoff)
               x[ind1] <- cutoff
               x[ind2] <- -cutoff
               V0[i,] <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
            }
            image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE,
               main=paste("Top Matches. Group:", h), sub = "", xlab= "", ylab="")
         }

         if (lens[h] == 1) {
           axis(2, at=1, labels=rev(r.names), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
                line=-0.7, font=2, family="")
           axis(4, at=1+0.4, labels=" CIC     p-val     FDR", adj= 0.5, tick=FALSE, las = 1,
                cex.axis=0.8*character.scaling, line=-0.7, font=2, family="")
           axis(4, at=1, labels=all.vals0, adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
                line=-0.7, font=2, family="")
         } else {
            axis(2, at=1:nrow(V0), labels=rev(r.names), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
                 line=-0.7, font=2, family="")
            axis(4, at=nrow(V0)+0.4, labels=" CIC     p-val     FDR", adj= 0.5, tick=FALSE, las = 1,
                 cex.axis=0.8*character.scaling, line=-0.7, font=2, family="")            
            axis(4, at=c(seq(1, nrow(V0) - 1), nrow(V0) - 0.2), labels=rev(all.vals0), adj= 0.5,
                 tick=FALSE, las = 1, cex.axis=character.scaling, line=-0.7, font=2, family="")
         }
      }
      par(def.par)
      
      # Update seed

      seed.names.iter[iter] <- cmi.names[1, iter] # top hit from this iteration
      seed <- apply(rbind(seed, m.2[seed.names.iter[iter],]), MARGIN=2, FUN=seed.combination.op)
      seed.iter[iter,] <- seed
      cmi.seed[iter] <- CCBA_assoc.v1(target, seed, assoc.metric)
      pct_explained.seed[iter] <- sum(seed[target.locs])/length(target.locs)
      
    } # end of iterations loop

   # Final summary figures -----------------------------------------------------------------------------

   summ.panel <- length(seed.names) + 2 * max.n.iter + 2
  
   legend.size <- 4
   pad.space <- 30 - summ.panel - legend.size

   nf <- layout(matrix(c(1, 2, 3, 0), 4, 1, byrow=T), 1, c(2, summ.panel, legend.size, pad.space), FALSE)

   cutoff <- 2.5
   x <- as.numeric(target)         
   x <- (x - mean(x))/sd(x)         
   ind1 <- which(x > cutoff)
   ind2 <- which(x < -cutoff)
   x[ind1] <- cutoff
   x[ind2] <- -cutoff
   V1 <- ceiling(ncolors * (x + cutoff)/(cutoff*2))

   par(mar = c(1, 22, 2, 12))
   image(1:length(target), 1:1, as.matrix(V1), zlim=c(0, ncolors), col=mycol, axes=FALSE,
         main=paste("REVEALER - Results"), sub = "", xlab= "", ylab="", font=2, family="")
  
   axis(2, at=1:1, labels=paste("TARGET:  ", target.name), adj= 0.5, tick=FALSE,las = 1, cex=1,
        cex.axis=character.scaling,  line=0, font=2, family="")

   axis(4, at=1:1, labels="  IC ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="black")
   axis(4, at=1:1, labels="       / CIC", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="steelblue")

#   axis(4, at=1:1, labels="  IC   ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")

   V0 <- rbind(seed.vectors + 2, seed.cum) 
   for (i in 1:max.n.iter) {
      V0 <- rbind(V0,
                  m.2[seed.names.iter[i],] + 2,
                  seed.iter[i,])
   }

   row.names.V0 <- c(paste("SEED:   ", seed.names), "SUMMARY SEED:")
   for (i in 1:max.n.iter) {
      row.names.V0 <- c(row.names.V0, paste("ITERATION ", i, ":  ", seed.names.iter[i], sep=""),
                        paste("SUMMARY FEATURE ", i, ":  ", sep=""))
   }
   row.names(V0) <- row.names.V0

   cmi.vals <- c(cmi.orig.seed, cmi.orig.seed.cum[length(seed.names)])                 
   for (i in 1:max.n.iter) {
      cmi.vals <- c(cmi.vals, as.vector(cmi[1, i]), cmi.seed[i])
    }
   cmi.vals <- signif(cmi.vals, 2)
   all.vals <-cmi.vals

   cmi.cols <- c(rep("black", length(cmi.orig.seed)), "black", rep(c("steelblue", "black"), max.n.iter))                     # IC/CIC colors   

   V0 <- apply(V0, MARGIN=2, FUN=rev)

   par(mar = c(7, 22, 0, 12))
   if (feature.type == "discrete") {  
       image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 3),
       col=c(brewer.pal(9, "Greys")[2], brewer.pal(9, "Greys")[5],                          
             brewer.pal(9, "Blues")[3], brewer.pal(9, "Blues")[9]), axes=FALSE, main="",
             sub = "", xlab= "", ylab="")
   } else { # continuous
      for (i in 1:nrow(V0)) {
         cutoff <- 2.5
         x <- as.numeric(V0[i,])
         x <- (x - mean(x))/sd(x)         
         ind1 <- which(x > cutoff)
         ind2 <- which(x < -cutoff)
         x[ind1] <- cutoff
         x[ind2] <- -cutoff
         V0[i,] <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
      }
      image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE, main="",
            sub = "", xlab= "", ylab="")
   }
#   axis(2, at=1:nrow(V0), labels=row.names(V0), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")
#   axis(4, at=1:nrow(V0), labels=rev(all.vals), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")
    for (axis.i in 1:nrow(V0)) {
          axis(2, at=axis.i, labels=row.names(V0)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
          axis(4, at=axis.i, labels=rev(all.vals)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
      }
   
   axis(1, at=1:ncol(V0), labels=colnames(V0), adj= 0.5, tick=FALSE,las = 3, cex=1, cex.axis=0.4*character.scaling,
        line=0, font=2, family="")

        # Legend

      par.mar <- par("mar")
      par(mar = c(3, 35, 8, 10))
      leg.set <- seq(-cutoff, cutoff, 2*cutoff/100)
      image(1:101, 1:1, as.matrix(leg.set), zlim=c(-cutoff, cutoff), col=mycol, axes=FALSE, main="",
          sub = "", xlab= "", ylab="",font=2, family="")
      ticks <- c(-2, -1, 0, 1, 2)
      tick.cols <- rep("black", 5)
      tick.lwd <- c(1,1,2,1,1)
      locs <- NULL
      for (k in 1:length(ticks)) locs <- c(locs, which.min(abs(ticks[k] - leg.set)))
      axis(1, at=locs, labels=ticks, adj= 0.5, tick=T, cex=0.8, cex.axis=1, line=0, font=2, family="")
      mtext("Standardized Target Profile", cex=0.8, side = 1, line = 3.5, outer=F)
      par(mar = par.mar)

   V0 <- rbind(target, seed.vectors, seed.cum) 
   for (i in 1:max.n.iter) {
      V0 <- rbind(V0,
                  m.2[seed.names.iter[i],],
                  seed.iter[i,])
   }
   V0.colnames <- colnames(V0)
   V0 <- cbind(V0, c(1, all.vals))
   colnames(V0) <- c(V0.colnames, "IC")

   row.names.V0 <- c(target.name, seed.names, "SUMMARY SEED:")
   for (i in 1:max.n.iter) {
      row.names.V0 <- c(row.names.V0, seed.names.iter[i], paste("SUMMARY FEATURE ", i, ":  ", sep=""))
   }
   row.names(V0) <- row.names.V0
  
  # Version without summaries ----------------------------------------------------

   summ.panel <- length(seed.names) + max.n.iter + 2
  
   legend.size <- 4
   pad.space <- 30 - summ.panel - legend.size
   
   nf <- layout(matrix(c(1, 2, 3, 0), 4, 1, byrow=T), 1, c(2, summ.panel, legend.size, pad.space), FALSE)

   cutoff <- 2.5
   x <- as.numeric(target)         
   x <- (x - mean(x))/sd(x)         
   ind1 <- which(x > cutoff)
   ind2 <- which(x < -cutoff)
   x[ind1] <- cutoff
   x[ind2] <- -cutoff
   V1 <- ceiling(ncolors * (x + cutoff)/(cutoff*2))

   par(mar = c(1, 22, 2, 12))
   image(1:length(target), 1:1, as.matrix(V1), zlim=c(0, ncolors), col=mycol, axes=FALSE,
         main=paste("REVEALER - Results"),
         sub = "", xlab= "", ylab="", font=2, family="")
  
   axis(2, at=1:1, labels=paste("TARGET:  ", target.name), adj= 0.5, tick=FALSE,las = 1, cex=1,
        cex.axis=character.scaling,  line=0, font=2, family="")
   
#   axis(4, at=1:1, labels="  IC   ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")

      axis(4, at=1:1, labels="  IC ", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="black")
      axis(4, at=1:1, labels="       / CIC", adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,     # IC/CIC
           font.axis=1, line=0, font=2, family="", col.axis="steelblue")

   
   V0 <- seed.vectors + 2
   for (i in 1:max.n.iter) {
      V0 <- rbind(V0,
                  m.2[seed.names.iter[i],] + 2)
   }
   V0 <- rbind(V0, seed.iter[max.n.iter,])

   row.names.V0 <- c(paste("SEED:   ", seed.names))
   for (i in 1:max.n.iter) {
      row.names.V0 <- c(row.names.V0, paste("ITERATION ", i, ":  ", seed.names.iter[i], sep=""))
   }
   row.names(V0) <- c(row.names.V0, "FINAL SUMMARY")

   cmi.vals <- cmi.orig.seed
   for (i in 1:max.n.iter) {
      cmi.vals <- c(cmi.vals, as.vector(cmi[1, i]))
    }
   cmi.vals <- c(cmi.vals, cmi.seed[max.n.iter])
   cmi.vals <- signif(cmi.vals, 2)
   all.vals <-cmi.vals

   cmi.cols <- c(rep("black", length(cmi.orig.seed)), rep("steelblue", max.n.iter), "black")    # IC/CIC colors   
      
   V0 <- apply(V0, MARGIN=2, FUN=rev)
   par(mar = c(7, 22, 0, 12))   
   
   if (feature.type == "discrete") {  
       image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, 3),
       col=c(brewer.pal(9, "Greys")[2], brewer.pal(9, "Greys")[5],                          
              brewer.pal(9, "Blues")[3], brewer.pal(9, "Blues")[9]), axes=FALSE, main="",
             sub = "", xlab= "", ylab="")
   } else { # continuous
      for (i in 1:nrow(V0)) {
         cutoff <- 2.5
         x <- as.numeric(V0[i,])
         x <- (x - mean(x))/sd(x)         
         ind1 <- which(x > cutoff)
         ind2 <- which(x < -cutoff)
         x[ind1] <- cutoff
         x[ind2] <- -cutoff
         V0[i,] <- ceiling(ncolors * (x + cutoff)/(cutoff*2))
      }
      image(1:ncol(V0), 1:nrow(V0), t(V0), zlim = c(0, ncolors), col=mycol, axes=FALSE, main="",
            sub = "", xlab= "", ylab="")
   }
#   axis(2, at=1:nrow(V0), labels=row.names(V0), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")
#   axis(4, at=1:nrow(V0), labels=rev(all.vals), adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
#        line=0, font=2, family="")
      for (axis.i in 1:nrow(V0)) {
          axis(2, at=axis.i, labels=row.names(V0)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
          axis(4, at=axis.i, labels=rev(all.vals)[axis.i], adj= 0.5, tick=FALSE, las = 1, cex.axis=character.scaling,
               line=0, font=2, family="", col.axis = rev(cmi.cols)[axis.i])
      }
   
   axis(1, at=1:ncol(V0), labels=colnames(V0), adj= 0.5, tick=FALSE,las = 3, cex=1, cex.axis=0.4*character.scaling,
        line=0, font=2, family="")

        # Legend

      par.mar <- par("mar")
      par(mar = c(3, 35, 8, 10))
      leg.set <- seq(-cutoff, cutoff, 2*cutoff/100)
      image(1:101, 1:1, as.matrix(leg.set), zlim=c(-cutoff, cutoff), col=mycol, axes=FALSE, main="",
          sub = "", xlab= "", ylab="",font=2, family="")
      ticks <- c(-2, -1, 0, 1, 2)
      tick.cols <- rep("black", 5)
      tick.lwd <- c(1,1,2,1,1)
      locs <- NULL
      for (k in 1:length(ticks)) locs <- c(locs, which.min(abs(ticks[k] - leg.set)))
      axis(1, at=locs, labels=ticks, adj= 0.5, tick=T, cex=0.8, cex.axis=0.8, line=0, font=2, family="")
      mtext("Standardized Target Profile", cex=0.8, side = 1, line = 3.5, outer=F)
      par(mar = par.mar)

  # Landscape plot ---------------------------------------------------------------

  if (produce.lanscape.plot == T) {
  
   nf <- layout(matrix(c(1, 2), 2, 1, byrow=T), 1, c(2, 1), FALSE)

    if (length(as.vector(cmi.names[1:top.n, 1:max.n])) > 1) {
       V0 <- rbind(seed.vectors, as.matrix(m.2[as.vector(cmi.names[1:top.n, 1:max.n]),]))
    } else {
       V0 <- rbind(seed.vectors, t(as.matrix(m.2[as.vector(cmi.names[1:top.n, 1:max.n]),])))
    }
   
   number.seq <- NULL
   for (i in 1:max.n) number.seq <- c(number.seq, rep(i, top.n))

   row.names(V0) <-  c(paste("SEED: ", seed.names, "(", signif(cmi.orig.seed, 2), ")"),
                       paste("ITER ", number.seq, ":", as.vector(cmi.names[1:top.n, 1:max.n]),
                             "(", signif(as.vector(cmi[1:top.n, 1:max.n]), 2), ")"))
 
    cmi.vals <- c(cmi.orig.seed, as.vector(cmi[1:top.n, 1:max.n]))

   total.points <- row(V0)
   V2 <- V0
   metric.matrix <- matrix(0, nrow=nrow(V2), ncol=nrow(V2))
   row.names(metric.matrix)  <- row.names(V2)
   colnames(metric.matrix) <- row.names(V2)
   MI.ref <- cmi.vals
   for (i in 1:nrow(V2)) {
      for (j in 1:i) {
           metric.matrix[i, j] <- CCBA_assoc.v1(V2[j,], V2[i,], assoc.metric)
      }
   }
   metric.matrix
   metric.matrix <- metric.matrix + t(metric.matrix)
   metric.matrix
   alpha <- 5
   metric.matrix2 <- 1 - ((1/(1+exp(-alpha*metric.matrix))))
   for (i in 1:nrow(metric.matrix2)) metric.matrix2[i, i] <- 0
   metric.matrix2
 
   smacof.map <- smacofSphere(metric.matrix2, ndim = 2, weightmat = NULL, init = NULL,
                                     ties = "primary", verbose = FALSE, modulus = 1, itmax = 1000, eps = 1e-6)
   x0 <- smacof.map$conf[,1]
   y0 <- smacof.map$conf[,2]
   r <- sqrt(x0*x0 + y0*y0)
   radius <-  1 - ((1/(1+exp(-alpha*MI.ref))))
   x <- x0*radius/r
   y <- y0*radius/r
   angles <- atan2(y0, x0)
   
   par(mar = c(4, 7, 4, 7))
 
   plot(x, y, pch=20, bty="n", xaxt='n', axes = FALSE, type="n", xlab="", ylab="",
        main=paste("REVEALER - Landscape for ", target.name),
        xlim=1.2*c(-max(radius), max(radius)), ylim=1.2*c(-max(radius), max(radius)))
   line.angle <- seq(0, 2*pi-0.001, 0.001)
   for (i in 1:length(x)) {
      line.max.x <- radius[i] * cos(line.angle)
      line.max.y <- radius[i] * sin(line.angle)
      points(line.max.x, line.max.y, type="l", col="gray80", lwd=1)
      points(c(0, x[i]), c(0, y[i]), type="l", col="gray80", lwd=1)
   }
   line.max.x <- 1.2*max(radius) * cos(line.angle)
   line.max.y <- 1.2*max(radius) * sin(line.angle)
   points(line.max.x, line.max.y, type="l", col="purple", lwd=2)
   points(0, 0, pch=21, bg="red", col="black", cex=2.5)   
   points(x, y, pch=21, bg="steelblue", col="black", cex=2.5)

   x <- c(0, x)
   y <- c(0, y)
 
   text(x[1], y[1], labels=print.names[1], pos=2, cex=0.85, col="red", offset=1, font=2, family="")   
   for (i in 2:length(x)) {
      pos <- ifelse(x[i] <= 0.25, 4, 2)
     text(x[i], y[i], labels=print.names[i], pos=pos, cex=0.50, col="darkblue", offset=1, font=2, family="")   
    }

  }
   dev.off()
}


#-------------------------------------------------------------------------------------------------
   CCBA_assoc.v1 <- function(
   #
   # Pairwise association of x and y
   # P. Tamayo Jan 17, 2016

    x,
    y,
    metric) { 
    if (length(unique(x)) == 1 || length(unique(y)) == 1) return(0)
    if (metric == "IC") {
       return(CCBA_mutual.inf.v2(x = x, y = y, n.grid=25)$IC)
    } else if (metric == "COR") {
        return(cor(x, y))
    }
   }

#-------------------------------------------------------------------------------------------------
   CCBA_cond.assoc.v1 <-  function(
   #
   # Association of a and y given z
   # P. Tamayo Jan 17, 2016       
   #
    x,
    y,
    z,
    metric = "IC")
    { 

       if (length(unique(x)) == 1 || length(unique(y)) == 1) return(0)

       if (length(unique(z)) == 1) {  # e.g. for NULLSEED
          if (metric == "IC") {
             return(CCBA_mutual.inf.v2(x = x, y = y, n.grid = 25)$IC)
          } else if (metric == "COR") {
             return(cor(x, y))
          }
      } else {
          if (metric == "IC") {
             return(CCBA_cond.mutual.inf.v1(x = x, y = y, z = z, n.grid = 25)$CIC)
          } else if (metric == "COR") {
             return(pcor.test(x, y, z)$estimate)
          }
      }
   }

#-------------------------------------------------------------------------------------------------
    CCBA_cond.mutual.inf.v1 <- function(
    #
    # Computes the Conditional mutual imnformation: 
    #  I(X, Y | X) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    # The 0.25 in front of the bandwidth is because different conventions between bcv and kde3d
    # P. Tamayo Jan 17, 2016
        
    x,
    y,
    z,
    n.grid  =  25,
    delta   =  0.25*c(bcv(x), bcv(y), bcv(z)))
   {
        
   rho <- cor(x, y)
   rho2 <- ifelse(rho < 0, 0, rho)
   delta <- delta*(1 + (-0.75)*rho2)
  
   kde3d.xyz <- kde3d(x=x, y=y, z=z, h=delta, n = n.grid)
   X <- kde3d.xyz$x
   Y <- kde3d.xyz$y
   Z <- kde3d.xyz$z
   PXYZ <- kde3d.xyz$d + .Machine$double.eps

   # get grid spacing
   dx <- X[2] - X[1]
   dy <- Y[2] - Y[1]
   dz <- Z[2] - Z[1]

   # normalize density and calculate marginal densities and entropies
   PXYZ <- PXYZ/(sum(PXYZ)*dx*dy*dz)
   PXZ <- colSums(aperm(PXYZ, c(2,1,3)))*dy
   PYZ <- colSums(PXYZ)*dx
   PZ <- rowSums(aperm(PXYZ, c(3,1,2)))*dx*dy
   PXY <- colSums(aperm(PXYZ, c(3,1,2)))*dz
   PX <- rowSums(PXYZ)*dy*dz
   PY <- rowSums(aperm(PXYZ, c(2,1,3)))*dx*dz
   
   HXYZ <- - sum(PXYZ * log(PXYZ))*dx*dy*dz
   HXZ <- - sum(PXZ * log(PXZ))*dx*dz
   HYZ <- - sum(PYZ * log(PYZ))*dy*dz
   HZ <-  - sum(PZ * log(PZ))*dz
   HXY <- - sum(PXY * log(PXY))*dx*dy   
   HX <-  - sum(PX * log(PX))*dx
   HY <-  - sum(PY * log(PY))*dy

   MI <- HX + HY - HXY   
   CMI <- HXZ + HYZ - HXYZ - HZ

   SMI <- sign(rho) * MI
   SCMI <- sign(rho) * CMI

   IC <- sign(rho) * sqrt(1 - exp(- 2 * MI))
   CIC <- sign(rho) * sqrt(1 - exp(- 2 * CMI))
   
   return(list(CMI=CMI, MI=MI, SCMI=SCMI, SMI=SMI, HXY=HXY, HXYZ=HXYZ, IC=IC, CIC=CIC))
 }

#-------------------------------------------------------------------------------------------------
    CCBA_mutual.inf.v2 <- function(
    #
    # REVEALER IC coefficient
    # For definitions of mutual information and the universal metric (NMI) see the 
    # definition of "Mutual Information" in wikipedia and Thomas and Cover's book
    # P. Tamayo Jan 17, 2016
    #
    x,
    y,
    n.grid=25,
    delta = c(bcv(x), bcv(y))) 
    {
   rho <- cor(x, y)
   rho2 <- abs(rho)
   delta <- delta*(1 + (-0.75)*rho2)

   kde2d.xy <- kde2d(x, y, n = n.grid, h = delta)
   FXY <- kde2d.xy$z + .Machine$double.eps
   dx <- kde2d.xy$x[2] - kde2d.xy$x[1]
   dy <- kde2d.xy$y[2] - kde2d.xy$y[1]
   PXY <- FXY/(sum(FXY)*dx*dy)
   PX <- rowSums(PXY)*dy
   PY <- colSums(PXY)*dx
   HXY <- -sum(PXY * log(PXY))*dx*dy
   HX <- -sum(PX * log(PX))*dx
   HY <- -sum(PY * log(PY))*dy

   PX <- matrix(PX, nrow=n.grid, ncol=n.grid)
   PY <- matrix(PY, byrow = TRUE, nrow=n.grid, ncol=n.grid)

   MI <- sum(PXY * log(PXY/(PX*PY)))*dx*dy
   rho <- cor(x, y)
   SMI <- sign(rho) * MI
   
   IC <- sign(rho) * sqrt(1 - exp(- 2 * MI)) 
   
   NMI <- sign(cor(x, y)) * ((HX + HY)/HXY - 1)  # use peason correlation the get the sign (directionality)

   return(list(MI=MI, SMI=SMI, HXY=HXY, HX=HX, HY=HY, NMI=NMI, IC=IC))
}

#-------------------------------------------------------------------------------------------------
   CCBA_ssGSEA_project_dataset.v1 <- function(
   #
   # Project dataset into pathways or gene sets using ssGSEA
   # P. Tamayo Jan 17, 2016
   #
       input.ds,
       output.ds,
       gene.set.databases,
       gene.set.selection  = "ALL",   # "ALL" or list with names of gene sets
       sample.norm.type    = "rank",  # "rank", "log" or "log.rank"
       weight              = 0.25,
       statistic           = "area.under.RES",
       output.score.type   = "ES",    # "ES" or "NES"
       nperm               = 200,     # number of random permutations for NES case
       combine.mode        = "combine.off",  # "combine.off" do not combine *_UP and *_DN versions in 
		# a single score. "combine.replace" combine *_UP and 
		# *_DN versions in a single score that replaces the individual
		# *_UP and *_DN versions. "combine.add" combine *_UP and 
		# *_DN versions in a single score and add it but keeping 
		# the individual *_UP and *_DN versions.
       min.overlap         = 1,
       gene.names.in.desc  = F,      # in Protein, RNAi Ataris or hairpin gct files the gene symbols are in the descs column
       correl.type         = "rank") # "rank", "z.score", "symm.rank"
   { 
	
	# Read input dataset
	
	dataset <- CCBA_read_GCT_file.v1(filename = input.ds)  # Read gene expression dataset (GCT format)
	m <- data.matrix(dataset$ds)
        if (gene.names.in.desc == T) {
      	    gene.names <- dataset$descs
        } else {
            gene.names <- dataset$row.names
        }
	gene.descs <- dataset$descs
	sample.names <- dataset$names
	Ns <- length(m[1,])
	Ng <- length(m[,1])
	temp <- strsplit(input.ds, split="/") # Extract input file name
	s <- length(temp[[1]])
	input.file.name <- temp[[1]][s]
	temp <- strsplit(input.file.name, split=".gct")
	input.file.prefix <-  temp[[1]][1]
	
	# Sample normalization
	
	if (sample.norm.type == "rank") {
		for (j in 1:Ns) {  # column rank normalization 
			m[,j] <- rank(m[,j], ties.method = "average")
		}
		m <- 10000*m/Ng
	} else if (sample.norm.type == "log.rank") {
		for (j in 1:Ns) {  # column rank normalization 
			m[,j] <- rank(m[,j], ties.method = "average")
		}
		m <- log(10000*m/Ng + exp(1))
	} else if (sample.norm.type == "log") {
		m[m < 1] <- 1
		m <- log(m + exp(1))
	}
	
	# Read gene set databases
	
	max.G <- 0
	max.N <- 0
	for (gsdb in gene.set.databases) {
    	   GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
           max.G <- max(max.G, max(GSDB$size.G))
	   max.N <- max.N +  GSDB$N.gs
	}
	N.gs <- 0
	gs <- matrix("null", nrow=max.N, ncol=max.G)
	gs.names <- vector(length=max.N, mode="character")
	gs.descs <- vector(length=max.N, mode="character")
	size.G <- vector(length=max.N, mode="numeric")
	start <- 1
	for (gsdb in gene.set.databases) {
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		N.gs <- GSDB$N.gs 
		gs.names[start:(start + N.gs - 1)] <- GSDB$gs.names
		gs.descs[start:(start + N.gs - 1)] <- GSDB$gs.desc
		size.G[start:(start + N.gs - 1)] <- GSDB$size.G
		gs[start:(start + N.gs - 1), 1:max(GSDB$size.G)] <- GSDB$gs[1:N.gs, 1:max(GSDB$size.G)]
		start <- start + N.gs
	}
	N.gs <- max.N
	
	# Select desired gene sets
	
	if (gene.set.selection[1] == "ALL") {
            gene.set.selection <- unique(gs.names)
        } 

        locs <- match(gene.set.selection, gs.names)
        # print(rbind(gene.set.selection, locs))
	N.gs <- sum(!is.na(locs))
	if(N.gs > 1) { 
           gs <- gs[locs,]
	} else { 
           gs <- t(as.matrix(gs[locs,]))   # Force vector to matrix if only one gene set specified
        }
   	gs.names <- gs.names[locs]
 	gs.descs <- gs.descs[locs]
	size.G <- size.G[locs]

        # Check for redundant gene sets

        tab <- as.data.frame(table(gs.names))
        ind <- order(tab[, "Freq"], decreasing=T)
        tab <- tab[ind,]
        max.n <- max(10, length(gs.names))
        print(tab[1:max.n,])
        print(paste("Total gene sets:", length(gs.names)))
        print(paste("Unique gene sets:", length(unique(gs.names))))

        # Loop over gene sets
	
	score.matrix <- score.matrix.2 <- matrix(0, nrow=N.gs, ncol=Ns)
        print(paste("Size score.matrix:", dim(score.matrix)))
        print(paste("Size score.matrix.2:", dim(score.matrix.2)))                
	for (gs.i in 1:N.gs) {
		#browser()
		gene.set <- gs[gs.i, 1:size.G[gs.i]]
		gene.overlap <- intersect(gene.set, gene.names)
		print(paste(gs.i, "gene set:", gs.names[gs.i], " overlap=", length(gene.overlap)))
                if (length(gene.overlap) < min.overlap) { 
			score.matrix[gs.i, ] <- rep(NA, Ns)
                        print(paste("Size score.matrix:", dim(score.matrix)))                        
			next
		} else {
			gene.set.locs <- match(gene.overlap, gene.set)
			gene.names.locs <- match(gene.overlap, gene.names)
			msig <- m[gene.names.locs,]
			msig.names <- gene.names[gene.names.locs]
			if (output.score.type == "ES") {
				OPAM <- CCBA_ssGSEA.Projection.v1(data.array = m, gene.names = gene.names, n.cols = Ns, 
						n.rows = Ng, weight = weight, statistic = statistic,
						gene.set = gene.overlap, nperm = 1, correl.type = correl.type)
				score.matrix[gs.i,] <- as.matrix(t(OPAM$ES.vector))
                                print(paste("Size score.matrix:", dim(score.matrix)))                                
			} else if (output.score.type == "NES") {
				OPAM <- CCBA_ssGSEA.Projection.v1(data.array = m, gene.names = gene.names, n.cols = Ns, 
						n.rows = Ng, weight = weight, statistic = statistic,
						gene.set = gene.overlap, nperm = nperm, correl.type = correl.type)
				score.matrix[gs.i,] <- as.matrix(t(OPAM$NES.vector))
                                print(paste("Size score.matrix:", dim(score.matrix)))                                
			}
		}
	}

        
        locs <- !is.na(score.matrix[,1])
        print(paste("N.gs before overlap prunning:", N.gs))
        N.gs <- sum(locs)
        print(paste("N.gs after overlap prunning:", N.gs))
        if (nrow(score.matrix) == 1) {
           score.matrix <- as.matrix(t(score.matrix[locs,]))
        } else {
           score.matrix <- score.matrix[locs,]           
        }
        print(paste("Size score.matrix:", dim(score.matrix)))        
        gs.names <- gs.names[locs]
        gs.descs <- gs.descs[locs]

	initial.up.entries <- 0
	final.up.entries <- 0
	initial.dn.entries <- 0
	final.dn.entries <- 0
	combined.entries <- 0
	other.entries <- 0
	
	if (combine.mode == "combine.off") {
                if (nrow(score.matrix) == 1) {
		score.matrix.2 <- as.matrix(t(score.matrix))
                } else {
		score.matrix.2 <- score.matrix
                }
                print(paste("Size score.matrix.2:", dim(score.matrix.2)))                                

		gs.names.2 <- gs.names
		gs.descs.2 <- gs.descs
	} else if ((combine.mode == "combine.replace") || (combine.mode == "combine.add")) {
		score.matrix.2 <- NULL
		gs.names.2 <- NULL
		gs.descs.2 <- NULL
		k <- 1
		for (i in 1:N.gs) {
			temp <- strsplit(gs.names[i], split="_") 
			body <- paste(temp[[1]][seq(1, length(temp[[1]]) -1)], collapse="_")
			suffix <- tail(temp[[1]], 1)
			print(paste("i:", i, "gene set:", gs.names[i], "body:", body, "suffix:", suffix))
			if (suffix == "UP") {  # This is an "UP" gene set
				initial.up.entries <- initial.up.entries + 1
				target <- paste(body, "DN", sep="_")
				loc <- match(target, gs.names)            
				if (!is.na(loc)) { # found corresponding "DN" gene set: create combined entry
					score <- score.matrix[i,] - score.matrix[loc,]
					score.matrix.2 <- rbind(score.matrix.2, score)
					gs.names.2 <- c(gs.names.2, body)
					gs.descs.2 <- c(gs.descs.2, paste(gs.descs[i], "combined UP & DN"))
					combined.entries <- combined.entries + 1
					if (combine.mode == "combine.add") {  # also add the "UP entry
                                           if (nrow(score.matrix) == 1) {
	  				      score.matrix.2 <- rbind(score.matrix.2, as.matrix(t(score.matrix[i,])))
                                           } else {
				  	      score.matrix.2 <- rbind(score.matrix.2, score.matrix[i,])
                                           }
                                           print(paste("Size score.matrix.2:", dim(score.matrix.2)))
                                           gs.names.2 <- c(gs.names.2, gs.names[i])
					   gs.descs.2 <- c(gs.descs.2, gs.descs[i])
					   final.up.entries <- final.up.entries + 1
					}
				} else { # did not find corresponding "DN" gene set: create "UP" entry
                                        if (nrow(score.matrix) == 1) {
					   score.matrix.2 <- rbind(score.matrix.2, as.matrix(t(score.matrix[i,])))
                                        } else {
				  	   score.matrix.2 <- rbind(score.matrix.2, score.matrix[i,])
                                        }
                                        print(paste("Size score.matrix.2:", dim(score.matrix.2)))                        
					gs.names.2 <- c(gs.names.2, gs.names[i])
					gs.descs.2 <- c(gs.descs.2, gs.descs[i])
					final.up.entries <- final.up.entries + 1
				}
			} else if (suffix == "DN") { # This is a "DN" gene set
				initial.dn.entries <- initial.dn.entries + 1
				target <- paste(body, "UP", sep="_")
				loc <- match(target, gs.names)            
				if (is.na(loc)) { # did not find corresponding "UP" gene set: create "DN" entry
                                        if (nrow(score.matrix) == 1) {
					   score.matrix.2 <- rbind(score.matrix.2, as.matrix(t(score.matrix[i,])))
                                        } else {
				  	   score.matrix.2 <- rbind(score.matrix.2, score.matrix[i,])
                                        }
                                        print(paste("Size score.matrix.2:", dim(score.matrix.2)))                                        
					gs.names.2 <- c(gs.names.2, gs.names[i])
					gs.descs.2 <- c(gs.descs.2, gs.descs[i])
					final.dn.entries <- final.dn.entries + 1
				} else { # it found corresponding "UP" gene set
					if (combine.mode == "combine.add") { # create "DN" entry
                                           if (nrow(score.matrix) == 1) {
				   	      score.matrix.2 <- rbind(score.matrix.2, as.matrix(t(score.matrix[i,])))
                                           } else {
				  	      score.matrix.2 <- rbind(score.matrix.2, score.matrix[i,])
                                           }
                                           print(paste("Size score.matrix.2:", dim(score.matrix.2)))                                           
					   gs.names.2 <- c(gs.names.2, gs.names[i])
					   gs.descs.2 <- c(gs.descs.2, gs.descs[i])
					   final.dn.entries <- final.dn.entries + 1
					}
				}
			} else { # This is neither "UP nor "DN" gene set: create individual entry
                                     if (nrow(score.matrix) == 1) {
				   	 score.matrix.2 <- rbind(score.matrix.2, as.matrix(t(score.matrix[i,])))
                                      } else {
				  	 score.matrix.2 <- rbind(score.matrix.2, score.matrix[i,])
                                      }
                                      print(paste("Size score.matrix.2:", dim(score.matrix.2)))
                                      gs.names.2 <- c(gs.names.2, gs.names[i])
				      gs.descs.2 <- c(gs.descs.2, gs.descs[i])
				      other.entries <- other.entries + 1
			}
		} # end for loop over gene sets
		print(paste("initial.up.entries:", initial.up.entries))
		print(paste("final.up.entries:", final.up.entries))
		print(paste("initial.dn.entries:", initial.dn.entries))
		print(paste("final.dn.entries:", final.dn.entries))
		print(paste("other.entries:", other.entries))
		print(paste("combined.entries:", combined.entries))

		print(paste("total entries:", length(score.matrix.2[,1])))
	}            

       # Make sure there are no duplicated gene names after adding entries
        
        unique.gene.sets <- unique(gs.names.2)
        locs <- match(unique.gene.sets, gs.names.2)
        if (nrow(score.matrix) == 1) {
           score.matrix.2 <- as.matrix(t(score.matrix.2[locs,]))
       } else {
           score.matrix.2 <- score.matrix.2[locs,]
       }

        gs.names.2 <- gs.names.2[locs]
        gs.descs.2 <- gs.descs.2[locs]
        
        # Final count

        tab <- as.data.frame(table(gs.names.2))
        ind <- order(tab[, "Freq"], decreasing=T)
        tab <- tab[ind,]
        print(tab[1:20,])
        print(paste("Total gene sets:", length(gs.names.2)))
        print(paste("Unique gene sets:", length(unique(gs.names.2))))
        
	V.GCT <- data.frame(score.matrix.2)
	colnames(V.GCT) <- sample.names
	row.names(V.GCT) <- gs.names.2
	CCBA_write.gct.v1(gct.data.frame = V.GCT, descs = gs.descs.2, filename = output.ds)  
	
} 

#-------------------------------------------------------------------------------------------------
    CCBA_Read.GeneSets.db.v1 <- function(
    #
    # Read gene sets from a database (GMT file)
    # P. Tamayo Jan 17, 2016
    #
	gs.db,
	thres.min   = 2,
	thres.max   = 2000,
	gene.names  = NULL)
   {
	
	temp <- readLines(gs.db)
	max.Ng <- length(temp)
	temp.size.G <- vector(length = max.Ng, mode = "numeric") 
	for (i in 1:max.Ng) {
		temp.size.G[i] <- length(unlist(strsplit(temp[[i]], "\t"))) - 2
	}
	max.size.G <- max(temp.size.G)      
	gs <- matrix(rep("null", max.Ng*max.size.G), nrow=max.Ng, ncol= max.size.G)
	temp.names <- vector(length = max.Ng, mode = "character")
	temp.desc <- vector(length = max.Ng, mode = "character")
	gs.count <- 1
	for (i in 1:max.Ng) {
		gene.set.size <- length(unlist(strsplit(temp[[i]], "\t"))) - 2
		gs.line <- noquote(unlist(strsplit(temp[[i]], "\t")))
		gene.set.name <- gs.line[1] 
		gene.set.desc <- gs.line[2] 
		gene.set.tags <- vector(length = gene.set.size, mode = "character")
		for (j in 1:gene.set.size) {
			gene.set.tags[j] <- gs.line[j + 2]
		}
		if (is.null(gene.names)) {
			existing.set <- rep(TRUE, length(gene.set.tags))
		} else {
			existing.set <- is.element(gene.set.tags, gene.names)
		}
		set.size <- length(existing.set[existing.set == T])
		if ((set.size < thres.min) || (set.size > thres.max)) next
		temp.size.G[gs.count] <- set.size
		gs[gs.count,] <- c(gene.set.tags[existing.set], rep("null", max.size.G - temp.size.G[gs.count]))
		temp.names[gs.count] <- gene.set.name
		temp.desc[gs.count] <- gene.set.desc
		gs.count <- gs.count + 1
	}
	Ng <- gs.count - 1
	gs.names <- vector(length = Ng, mode = "character")
	gs.desc <- vector(length = Ng, mode = "character")
	size.G <- vector(length = Ng, mode = "numeric") 
	
	gs.names <- temp.names[1:Ng]
	gs.desc <- temp.desc[1:Ng]
	size.G <- temp.size.G[1:Ng]
	
	return(list(N.gs = Ng, gs = gs, gs.names = gs.names, gs.desc = gs.desc, size.G = size.G, max.N.gs = max.Ng))
}

#-------------------------------------------------------------------------------------------------
   CCBA_ssGSEA.Projection.v1 <- function(
   #
   # ssGSEA projection
   # P. Tamayo Jan 17, 2016
   #
   # Runs a 2-3x faster (2-2.5x for ES statistic and 2.5-3x faster for area.under.ES statsitic)
   # version of GSEA.EnrichmentScore.5 internally that avoids overhead from the function call.
   # This function use dto be OPAM.Projection.3
       
	data.array,
	gene.names,
	n.cols,
	n.rows,
	weight = 0,
	statistic    = "Kolmogorov-Smirnov",  # "Kolmogorov-Smirnov", # "Kolmogorov-Smirnov", "Cramer-von-Mises",
	                                      # "Anderson-Darling", "Zhang_A", "Zhang_C", "Zhang_K",
 	                                      # "area.under.RES", or "Wilcoxon"
	gene.set,
	nperm = 200,
	correl.type  = "rank")                # "rank", "z.score", "symm.rank"
    {
	
	ES.vector <- vector(length=n.cols)
	NES.vector <- vector(length=n.cols)
	p.val.vector <- vector(length=n.cols)
	correl.vector <- vector(length=n.rows, mode="numeric")
	
   # Compute ES score for signatures in each sample
	
   #   print("Computing GSEA.....")
	phi <- array(0, c(n.cols, nperm))
	for (sample.index in 1:n.cols) {
		gene.list <- order(data.array[, sample.index], decreasing=T)            
		
		#      print(paste("Computing observed enrichment for UP signature in sample:", sample.index, sep=" ")) 

		gene.set2 <- match(gene.set, gene.names)
		
		if (weight == 0) {
			correl.vector <- rep(1, n.rows)
		} else if (weight > 0) {
			if (correl.type == "rank") {
				correl.vector <- data.array[gene.list, sample.index]
			} else if (correl.type == "symm.rank") {
				correl.vector <- data.array[gene.list, sample.index]
				correl.vector <- ifelse(correl.vector > correl.vector[ceiling(n.rows/2)], 
						correl.vector,
						correl.vector + correl.vector - correl.vector[ceiling(n.rows/2)]) 
			} else if (correl.type == "z.score") {
				x <- data.array[gene.list, sample.index]
				correl.vector <- (x - mean(x))/sd(x)
			}
		}
		### Olga's Additions ###
#		ptm.new = proc.time()
		tag.indicator <- sign(match(gene.list, gene.set2, nomatch=0))    # notice that the sign is 0 (no tag) or 1 (tag) 
		no.tag.indicator <- 1 - tag.indicator 
		N <- length(gene.list) 
		Nh <- length(gene.set2) 
		Nm <-  N - Nh 
		orig.correl.vector <- correl.vector
		if (weight == 0) correl.vector <- rep(1, N)   # unweighted case
		ind = which(tag.indicator==1)
		correl.vector <- abs(correl.vector[ind])^weight
		
		
		sum.correl = sum(correl.vector)
		up = correl.vector/sum.correl     # "up" represents the peaks in the mountain plot
		gaps = (c(ind-1, N) - c(0, ind))  # gaps between ranked pathway genes
		down = gaps/Nm
		
		RES = cumsum(c(up,up[Nh])-down)
		valleys = RES[1:Nh]-up
		
		max.ES = max(RES)
		min.ES = min(valleys)
		
		if( statistic == "Kolmogorov-Smirnov" ){
			if( max.ES > -min.ES ){
				ES <- signif(max.ES, digits=5)
				arg.ES <- which.max(RES)
			} else{
				ES <- signif(min.ES, digits=5)
				arg.ES <- which.min(RES)
			}
		}
		
		if( statistic == "area.under.RES"){
			if( max.ES > -min.ES ){
				arg.ES <- which.max(RES)
			} else{
				arg.ES <- which.min(RES)
			}
			gaps = gaps+1
			RES = c(valleys,0) * (gaps) + 0.5*( c(0,RES[1:Nh]) - c(valleys,0) ) * (gaps)
			ES = sum(RES)
		}
		GSEA.results = list(ES = ES, arg.ES = arg.ES, RES = RES, indicator = tag.indicator)
#		new.time <<- new.time + (proc.time() - ptm.new)
		### End Olga's Additions ###
		#GSEA.results <- GSEA.EnrichmentScore5(gene.list=gene.list, gene.set=gene.set2,
		#		statistic = statistic, alpha = weight, correl.vector = correl.vector)
		ES.vector[sample.index] <- GSEA.results$ES
		
		if (nperm == 0) {
			NES.vector[sample.index] <- ES.vector[sample.index]
			p.val.vector[sample.index] <- 1
		} else {
			for (r in 1:nperm) {
				reshuffled.gene.labels <- sample(1:n.rows)
				if (weight == 0) {
					correl.vector <- rep(1, n.rows)
				} else if (weight > 0) {
					correl.vector <- data.array[reshuffled.gene.labels, sample.index]
				} 
#				GSEA.results <- GSEA.EnrichmentScore5(gene.list=reshuffled.gene.labels, gene.set=gene.set2,
#						statistic = statistic, alpha = weight, correl.vector = correl.vector)
				### Olga's Additions ###
				tag.indicator <- sign(match(reshuffled.gene.labels, gene.set2, nomatch=0))    # notice that the sign is 0 (no tag) or 1 (tag) 
				no.tag.indicator <- 1 - tag.indicator 
				N <- length(reshuffled.gene.labels) 
				Nh <- length(gene.set2) 
				Nm <-  N - Nh 
#   orig.correl.vector <- correl.vector
				if (weight == 0) correl.vector <- rep(1, N)   # unweighted case
				ind <- which(tag.indicator==1)
				correl.vector <- abs(correl.vector[ind])^weight   
				
				sum.correl <- sum(correl.vector)
				up = correl.vector/sum.correl
				gaps = (c(ind-1, N) - c(0, ind))
				down = gaps/Nm
				
				RES = cumsum(c(up,up[Nh])-down)
				valleys = RES[1:Nh]-up
				
				max.ES = max(RES)
				min.ES = min(valleys)
				
				if( statistic == "Kolmogorov-Smirnov" ){
					if( max.ES > -min.ES ){
						ES <- signif(max.ES, digits=5)
						arg.ES <- which.max(RES)
					} else{
						ES <- signif(min.ES, digits=5)
						arg.ES <- which.min(RES)
					}
				}
				
				if( statistic == "area.under.RES"){
					if( max.ES > -min.ES ){
						arg.ES <- which.max(RES)
					} else{
						arg.ES <- which.min(RES)
					}
					gaps = gaps+1
					RES = c(valleys,0) * (gaps) + 0.5*( c(0,RES[1:Nh]) - c(valleys,0) ) * (gaps)
					ES = sum(RES)
				}
				
				GSEA.results = list(ES = ES, arg.ES = arg.ES, RES = RES, indicator = tag.indicator)
				### End Olga's Additions ###
				phi[sample.index, r] <- GSEA.results$ES
			}
			if (ES.vector[sample.index] >= 0) {
				pos.phi <- phi[sample.index, phi[sample.index, ] >= 0]
				if (length(pos.phi) == 0) pos.phi <- 0.5
				pos.m <- mean(pos.phi)
				NES.vector[sample.index] <- ES.vector[sample.index]/pos.m
				s <- sum(pos.phi >= ES.vector[sample.index])/length(pos.phi)
				p.val.vector[sample.index] <- ifelse(s == 0, 1/nperm, s)
			} else {
				neg.phi <-  phi[sample.index, phi[sample.index, ] < 0]
				if (length(neg.phi) == 0) neg.phi <- 0.5 
				neg.m <- mean(neg.phi)
				NES.vector[sample.index] <- ES.vector[sample.index]/abs(neg.m)
				s <- sum(neg.phi <= ES.vector[sample.index])/length(neg.phi)
				p.val.vector[sample.index] <- ifelse(s == 0, 1/nperm, s)
			}
		}
	}
	return(list(ES.vector = ES.vector, NES.vector =  NES.vector, p.val.vector = p.val.vector))
	
   } 

#-------------------------------------------------------------------------------------------------
    CCBA_OncoGPS_create_components.v1 <- function(
      #
      # Project an input dataset into components using NMF
      # P. Tamayo Jan 17, 2016
      #
      input_dataset,                    # Input GCT dataset A where the matrix decomposition takes place (A ~ W x H)
      input_normalization = "rank",     # Normalization for the input dataset: "rank"
      number_of_comp,                   # Number of components to use in the matrix decomposition
      prefix = "Comp",                  # Prefix for the component names
      method = "NMF",                   # Method for matrix factorization: NMF, snNMF or NMF_offset (IMF under construction)
      theta = 0.5,                      # For method = nsNMF value of smoothing parameter theta
      gene_subset = "all-genes",        # Universe of genes to consider for matrix decomposition: "gene-sets", "all-genes"
      gene_sets_files = NULL,           # If gene_subset = "gene-sets" GMT files with gene sets
      gene_sets = NULL,                 # If gene_subset = "gene-sets" then name of the specific gene set(s) in gene_sets_file to use
      normalize_after_selection = T,    # If gene_subset = "gene-sets," normalize after selection the gene subset
      preprojection_dataset = NULL,     # Save pre-projection input dataset in this file
      output_plots,                     # Output PDF file with W and H plots
      output_W_dataset,                 # Output GCT file with W matrix
      output_H_dataset,                 # Output GCT file with H matrix
      output_H_w_dataset,               # Output GCT file with W-derived H matrix
      output_A_dataset = NULL,          # Output GCT file with sorted and normalized A matrix
      row_classes_dataset = NULL,       # Output GCT with the row classes
      column_classes_dataset = NULL,    # Output GCT with the column classes
      seed = 123)                       # Randon number generator seed
  {
   set.seed(5209761)
   
   mycol <- vector(length=512, mode = "numeric")   # Red/Blue "pinkogram" color map
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)

   pdf(file=output_plots, height=8.5, width=11)

   comp.names <- paste(prefix, "C", seq(1, number_of_comp), "_", number_of_comp, sep="")
    
   # Read expression dataset

   dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   m.1 <- data.matrix(dataset.1$ds)
   print(paste("Dimensions matrix A:", nrow(m.1), ncol(m.1)))

  # heatmap(m.1, scale="row", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5, main="Sorted A Matrix", xlab = "Components", ylab= "Genes")   

   if (normalize_after_selection == F) {  # Normalize input data here before selection
     if (input_normalization == "rank") {

         max.n <- 10000
         for (i in 1:ncol(m.1)) m.1[,i] <- (max.n - 1) * (rank(m.1[,i]) - 1) /(nrow(m.1) - 1) + 1
   
      } else if (input_normalization == "none") {   

      } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
      }
   }
   
   if (gene_subset == "gene-sets") {  # select relevant genes from gene sets

        print("Selecting relevant genes from gene sets")

	max.G <- 0
	max.N <- 0
	for (gsdb in gene_sets_files) {
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		max.G <- max(max.G, max(GSDB$size.G))
		max.N <- max.N +  GSDB$N.gs
	}
	N.gs <- 0
	gs <- matrix("null", nrow=max.N, ncol=max.G)
	gs.names <- vector(length=max.N, mode="character")
	gs.descs <- vector(length=max.N, mode="character")
	size.G <- vector(length=max.N, mode="numeric")
	start <- 1
	for (gsdb in gene_sets_files) {  # Read all the gene sets from gene set files
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		N.gs <- GSDB$N.gs 
		gs.names[start:(start + N.gs - 1)] <- GSDB$gs.names
		gs.descs[start:(start + N.gs - 1)] <- GSDB$gs.desc
		size.G[start:(start + N.gs - 1)] <- GSDB$size.G
		gs[start:(start + N.gs - 1), 1:max(GSDB$size.G)] <- GSDB$gs[1:N.gs, 1:max(GSDB$size.G)]
		start <- start + N.gs
	}
	N.gs <- max.N
	
	# Select desired gene sets
	
	locs <- match(gene_sets, gs.names)
        print(rbind(gene_sets, locs))
	N.gs <- sum(!is.na(locs))
	if(N.gs > 1) { 
           gs <- gs[locs,]
	} else { 
           gs <- t(as.matrix(gs[locs,]))   # Force vector to matrix if only one gene set specified
        }
	gs.names <- gs.names[locs]
	gs.descs <- gs.descs[locs]
	size.G <- size.G[locs]

        genes <- NULL
       	for (gs.i in 1:N.gs) {
   	   gene.set <- gs[gs.i, 1:size.G[gs.i]]
           genes <- c(genes, gene.set)
         }
        print(paste("Number of selected genes:", length(genes)))
        genes <- unique(genes)
        print(paste("Number of unique selected genes:", length(genes)))        
        genes <- intersect(genes, row.names(m.1))
        print(paste("Number of overlapping genes (final set):", length(genes)))        
        m.2 <- m.1[genes,]
        print("Dimensions of selected input data:")
        print(dim(m.2))

   } else if (gene_subset == "all-genes") {

        print("Using all genes from gene sets")
        
      m.2 <- m.1
   } else {
      stop(paste("ERROR: unknown gene subset selection:", gene_subset))
   }

   if (normalize_after_selection == T) {  # Normalize input data here after selection
   
      if (input_normalization == "rank") {

         max.n <- 10000
         for (i in 1:ncol(m.2)) m.2[,i] <- (max.n - 1) * (rank(m.2[,i]) - 1) /(nrow(m.2) - 1) + 1

       } else if (input_normalization == "none") {   

       } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
       }
    }
   
   # Perform Matrix Factorization to find components

   if (!is.null(preprojection_dataset)) {
      CCBA_write.gct.v1(gct.data.frame = m.2, descs = row.names(m.2), preprojection_dataset)
    }

   if (method == "NMF") {
      NMF.out <- CCBA_NMF.div.v1(V = m.2, k = number_of_comp, maxniter = 1000, seed = seed, stopconv = 40, stopfreq = 10)
      W <- NMF.out$W
      H <- NMF.out$H
      row.names(W) <- row.names(m.2)
      colnames(W) <- row.names(H) <- comp.names
      colnames(H) <- colnames(m.2)

     plot(seq(1, length(NMF.out$error.v)), NMF.out$error.v, xlab="time", ylab="Error [divergence]", pch=20, col="blue")

      
   } else if (method == "IMF") {

    # To be added

   } else if (method == "NMF_offset") {

     NMF.out <- nmf(m.2, number_of_comp, "offset", seed=seed)
     W <- basis(NMF.out)
     H <- coef(NMF.out)
     row.names(W) <- row.names(m.2)
     colnames(W) <- row.names(H) <- comp.names
     colnames(H) <- colnames(m.2)

   } else if (method == "nsNMF") {

     NMF.out <- nmf(m.2, number_of_comp, "nsNMF", theta=theta, seed=seed)
     S <- nmfModel(number_of_comp, model='NMFns')
     S <- smoothing(S, theta=1)
     W <- basis(NMF.out) %*% S
     H <- coef(NMF.out)
     row.names(W) <- row.names(m.2)
     colnames(W) <- row.names(H) <- comp.names
     colnames(H) <- colnames(m.2)
     
   } else if (method == "NMF_fast") {

     NMF.out <- nmf(m.2, number_of_comp, "brunet", seed=seed)
     W <- basis(NMF.out)
     H <- coef(NMF.out)
     row.names(W) <- row.names(m.2)
     colnames(W) <- row.names(H) <- comp.names
     colnames(H) <- colnames(m.2)
   }
   
   # end of matrix factorization

  ind <- order(row.names(W))
  W <- W[ind,]
  m.2 <- m.2[ind,]
   
   # Obtain H via W: Project original and additional dataset using non-negative solver

   H_w <- matrix(0, nrow=number_of_comp, ncol= ncol(m.2), dimnames=list(row.names(H), colnames(H)))
   for (i in 1:ncol(H_w)) H_w[, i] <- nnls.fit(W, m.2[, i], wsqrt=1, eps=0, rank.tol=1e-07)

   # Save W and H matrices

   CCBA_write.gct.v1(gct.data.frame = W, descs = row.names(W), filename = output_W_dataset)
   CCBA_write.gct.v1(gct.data.frame = H, descs = row.names(H), filename = output_H_dataset)
   CCBA_write.gct.v1(gct.data.frame = H_w, descs = row.names(H_w), filename = output_H_w_dataset)   

   # Plot sorted W, H  and A matrices

   hc <- hclust(dist.IC(W, mode="cols"), method="ward") 
   d1.W <- as.dendrogram(hc)
   hc2 <- hclust(dist.IC(W, mode="rows"), method="ward") 
   d2.W <- as.dendrogram(hc2)
   heatmap(W, Colv=d1.W, Rowv = d2.W,  scale="row", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5,
           main="Sorted W Matrix", xlab = "Components", ylab= "Genes")

   row.classes <- cutree(hc2, k = number_of_comp)
   row.order <- order(row.classes, decreasing = F)
   row.classes <- row.classes[row.order]
   row.names <- row.names(W)[row.order]
   
   if (!is.null(row_classes_dataset)) {
      row.classes <- cbind(row.classes, row.classes)
      row.names(row.classes) <- row.names
      colnames(row.classes) <- c("ROW_classes", "ROW_classes2")      
      CCBA_write.gct.v1(gct.data.frame = row.classes, descs = row.names, filename = row_classes_dataset)
   }

   hc <- hclust(dist.IC(H, mode="cols"), method="ward") 
   d1.H <- as.dendrogram(hc)
   heatmap(H, Colv=d1.H, Rowv = d1.W,  scale="col", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5,
           main="Sorted H Matrix", xlab = "Components", ylab= "Genes")

   col.classes <- cutree(hc, k = number_of_comp)
   col.order <- order(col.classes, decreasing = F)
   col.classes <- col.classes[col.order]
   col.names <- col.names(H)[col.order]
   
   if (!is.null(column_classes_dataset)) {
      col.classes <- rbind(col.classes, col.classes)       
      row.names(col.classes) <- c("COL_classes", "COL_classes2")      
      colnames(col.classes) <- col.names
      CCBA_write.gct.v1(gct.data.frame = col.classes, descs = row.names(col.classes),
                  filename = column_classes_dataset)
   }

   hc <- hclust(dist.IC(H_w, mode="cols"), method="ward") 
   d1.H <- as.dendrogram(hc)
   heatmap(H_w, Colv=d1.H, Rowv = d1.W,  scale="col", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5,
           main="Sorted H_w Matrix", xlab = "Components", ylab= "Genes")

   hc <- hclust(dist.IC(m.2, mode="cols"), method="ward") 
   d1.A <- as.dendrogram(hc)
   hc2 <- hclust(dist.IC(m.2, mode="rows"), method="ward") 
   d2.A <- as.dendrogram(hc2)

#   heatmap(m.2, Colv=d1.A, Rowv = d2.A,  scale="none", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5,
#           main="Sorted A Matrix", xlab = "Components", ylab= "Genes")
    heatmap(m.2, scale="row", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5,
            main="Sorted A Matrix", xlab = "Components", ylab= "Genes")

   if (!is.null(output_A_dataset)) {
      m.2 <- m.2[hc2$order, hc$order]
      CCBA_write.gct.v1(gct.data.frame = m.2, descs = row.names(m.2), filename = output_A_dataset)
    }
   dev.off()
  
 }

#-------------------------------------------------------------------------------------------------	
   CCBA_OncoGPS_explore_component_creation.v1 <- function(
   #                                    
   #  Obtain statistics to choose number of components for Onco-GPS
   #  P. Tamayo Jan 17, 2016
   #
      input_dataset,                    # Input GCT dataset A where the matrix decomposition takes place (A ~ W x H)
      input_normalization = "rank",     # Normalization for the input dataset: "rank"
      k.min = 2,                        # Range of components: minimum
      k.max = 5,                        # Range of components: maximum
      k.incr = 1,                       # Range of components: increment                                         
      number_of_runs = 20,              # Number of runs to explore in the matrix decomposition
      method = "NMF",                   # Method for matrix factorization: NMF, nsNMF or NMF_offset (IMF under construction)
      theta = 0.5,                      # For method = nsNMF value of smoothing parameter theta       
      gene_subset = "all-genes",        # Universe of genes to consider for matrix decomposition: "gene-sets", "all-genes"
      gene_sets_files = NULL,           # If gene_subset = "gene-sets" GMT files with gene sets
      gene_sets = NULL,                 # If gene_subset = "gene-sets" then name of the specific gene set(s) in gene_sets_file to use
      normalize_after_selection = T,    # If gene_subset = "gene-sets," normalize after selection the gene subset
      output_plots)                     # Output PDF file with NMF plots
 {
   set.seed(5209761)
   
   mycol <- vector(length=512, mode = "numeric")   # Red/Blue "pinkogram" color map
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)
    
   # Read expression dataset

   dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   m.1 <- data.matrix(dataset.1$ds)
   print(paste("Dimensions matrix A:", nrow(m.1), ncol(m.1)))

  # heatmap(m.1, scale="row", col=mycol, margins=c(15, 15), cexRow=0.10, cexCol=0.5, main="Sorted A Matrix", xlab = "Components", ylab= "Genes")   

   if (normalize_after_selection == F) {  # Normalize input data here before selection
     if (input_normalization == "rank") {

         max.n <- 10000
         for (i in 1:ncol(m.1)) m.1[,i] <- (max.n - 1) * (rank(m.1[,i]) - 1) /(nrow(m.1) - 1) + 1
   
      } else if (input_normalization == "none") {   

      } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
      }
   }
   
   if (gene_subset == "gene-sets") {  # select relevant genes from gene sets

        print("Selecting relevant genes from gene sets")

	max.G <- 0
	max.N <- 0
	for (gsdb in gene_sets_files) {
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		max.G <- max(max.G, max(GSDB$size.G))
		max.N <- max.N +  GSDB$N.gs
	}
	N.gs <- 0
	gs <- matrix("null", nrow=max.N, ncol=max.G)
	gs.names <- vector(length=max.N, mode="character")
	gs.descs <- vector(length=max.N, mode="character")
	size.G <- vector(length=max.N, mode="numeric")
	start <- 1
	for (gsdb in gene_sets_files) {  # Read all the gene sets from gene set files
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		N.gs <- GSDB$N.gs 
		gs.names[start:(start + N.gs - 1)] <- GSDB$gs.names
		gs.descs[start:(start + N.gs - 1)] <- GSDB$gs.desc
		size.G[start:(start + N.gs - 1)] <- GSDB$size.G
		gs[start:(start + N.gs - 1), 1:max(GSDB$size.G)] <- GSDB$gs[1:N.gs, 1:max(GSDB$size.G)]
		start <- start + N.gs
	}
	N.gs <- max.N
	
	# Select desired gene sets
	
	locs <- match(gene_sets, gs.names)
        print(rbind(gene_sets, locs))
	N.gs <- sum(!is.na(locs))
	if(N.gs > 1) { 
           gs <- gs[locs,]
	} else { 
           gs <- t(as.matrix(gs[locs,]))   # Force vector to matrix if only one gene set specified
        }
	gs.names <- gs.names[locs]
	gs.descs <- gs.descs[locs]
	size.G <- size.G[locs]

        genes <- NULL
       	for (gs.i in 1:N.gs) {
   	   gene.set <- gs[gs.i, 1:size.G[gs.i]]
           genes <- c(genes, gene.set)
         }
        print(paste("Number of selected genes:", length(genes)))
        genes <- unique(genes)
        print(paste("Number of unique selected genes:", length(genes)))        
        genes <- intersect(genes, row.names(m.1))
        print(paste("Number of overlapping genes (final set):", length(genes)))        
        m.2 <- m.1[genes,]
        print("Dimensions of selected input data:")
        print(dim(m.2))

   } else if (gene_subset == "all-genes") {

        print("Using all genes from gene sets")
        
      m.2 <- m.1
   } else {
      stop(paste("ERROR: unknown gene subset selection:", gene_subset))
   }

   if (normalize_after_selection == T) {  # Normalize input data here after selection
   
      if (input_normalization == "rank") {

         max.n <- 10000
         for (i in 1:ncol(m.2)) m.2[,i] <- (max.n - 1) * (rank(m.2[,i]) - 1) /(nrow(m.2) - 1) + 1

       } else if (input_normalization == "none") {   

       } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
       }
    }
   
   # Perform Matrix Factorization to find components

   if (method == "NMF") {
     NMF.models <- nmf(m.2, seq(k.min, k.max + 1, k.incr), nrun = number_of_runs, method="brunet", seed=9876)
     
   } else if (method == "IMF") {

     # To be added

   } else if (method == "NMF_offset") {

     nmf(m.2, seq(k.min, k.max + 1, k.incr), nrun = number_of_runs, method="offset", seed=9876)
     
   } else if (method == "nsNMF") {

    nmf(m.2, seq(k.min, k.max + 1, k.incr), nrun = number_of_runs, method="nsNMF", theta=theta, seed=9876)

   }
   
#   quartz()
#   plot(NMF.models)

   NMF.sum <- summary(NMF.models)
   k.vec <- seq(k.min, k.max + 1, 1)
   cophen <- NMF.sum[, "cophenetic"]
   peak <- rep(0, length(k.vec))

   pdf(file=output_plots, height=8.5, width=11)

   plot(k.vec, cophen, type="n")
   points(k.vec, cophen, type="l")
   points(k.vec, cophen, type="p", pch=20)          

   for (h in 2:(length(cophen) - 1)) if (cophen[h - 1] < cophen[h] & cophen[h] > cophen[h + 1]) peak[h] <- 1
   k.peaks <- k.vec[peak == 1]
   k <- rev(k.peaks)[1]
   k
   print(paste("Suggested number of components:", k))

   consensusmap(NMF.models)
   
   # end of matrix factorization

   dev.off()

 }

#-------------------------------------------------------------------------------------------------	
    CCBA_NMF.div.v1 <- function(
    #
    # Non-Negative Matrix Factorization (NMF) 
    # P. Tamayo Jan 17, 2016
    #
        V,
        k,
        maxniter = 2000,
        seed     = 123456,
        stopconv = 40,
        stopfreq = 10)
       {

        N <- length(V[,1])
        M <- length(V[1,])
        set.seed(seed)
        W <- matrix(runif(N*k), nrow = N, ncol = k)  # Initialize W and H with random numbers
        H <- matrix(runif(k*M), nrow = k, ncol = M)
        VP <- matrix(nrow = N, ncol = M)
        error.v <- vector(mode = "numeric", length = maxniter)
        new.membership <- vector(mode = "numeric", length = M)
        old.membership <- vector(mode = "numeric", length = M)
        no.change.count <- 0
        eps <- .Machine$double.eps
        for (t in 1:maxniter) {
                VP = W %*% H
                W.t <- t(W)
                H <- H * (W.t %*% (V/VP)) + eps
                norm <- apply(W, MARGIN=2, FUN=sum)
                for (i in 1:k) {
                    H[i,] <- H[i,]/norm[i]
                }
                VP = W %*% H
                H.t <- t(H)
                W <- W * ((V/(VP + eps)) %*% H.t) + eps
                norm <- apply(H, MARGIN=1, FUN=sum)
                for (i in 1:k) {
                    W[,i] <- W[,i]/norm[i]
                }
               error.v[t] <- sum(V * log((V + eps)/(VP + eps)) - V + VP)/(M * N)
               if (t %% stopfreq == 0) {

                    for (j in 1:M) {
                        class <- order(H[,j], decreasing=T) 
                        new.membership[j] <- class[1]
                     }
                     if (sum(new.membership == old.membership) == M) {
                        no.change.count <- no.change.count + 1
                     } else {
                        no.change.count <- 0
                     }
                     if (no.change.count == stopconv) break
                     old.membership <- new.membership
               }
        }
        return(list(W = W, H = H, t = t, error.v = error.v))
}

#-------------------------------------------------------------------------------------------------	
    CCBA_subset_dataset_based_on_phenotype.v1 <- function(
    #
    # Subset a inpt dataset based on a phenotype defined in a table
    # P. Tamayo Jan 17, 2016
    #
    input_dataset,
    phen_table,
    phenotypes,
    output_dataset,
    exclude_these_samples = NULL)
   {
   dataset1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   m.1 <- data.matrix(dataset1$ds)
   sample.names.1 <- colnames(m.1)
   print(dim(m.1))

   if (!is.null(exclude_these_samples)) {
      locs <- match(exclude_these_samples, colnames(m.1))
      locs <- locs[!is.na(locs)]
      print(paste("Excluding samples:", sample.names.1[locs]))
      m.1 <- m.1[, -locs]
      sample.names.1 <- colnames(m.1)
      Ns.1 <- ncol(m.1)
      print(paste("Total samples after exclusion:", Ns.1))
    }  
   
    if (!is.null(phen_table) & !is.null(phenotypes)) {
         samples.table <- read.delim(phen_table, header=T, row.names=1, sep="\t", skip=0)
         print("colnames sample.table")
         print(colnames(samples.table))         
         
         table.sample.names <- row.names(samples.table)
         print("Subselecting samples with phenotypes: ")
         print(phenotypes)
         overlap <- intersect(colnames(m.1), table.sample.names)
         print(paste("overlap:", length(overlap)))
         locs1 <- match(overlap, table.sample.names)
         locs2 <- match(overlap, colnames(m.1))
         m.1 <- m.1[, locs2]
         sample.phen <- vector(length(locs1), mode="numeric")
         for (i in 1:length(locs1)) {
             sample.phen[i] <- 0
             for (j in 1:length(phenotypes)) {
                 print(paste("i:", i, " j:", j))
                 print(paste("locs1:", locs1[i]))
                 print(paste("phenotypes:", phenotypes[j]))                 
                 print(paste("phenotypes:", phenotypes[j]))                 
                 print(paste("samples.table[locs1[i], phenotypes[j]]:", samples.table[locs1[i], phenotypes[j]]))
                if (!is.na(samples.table[locs1[i], phenotypes[j]])) {
                    if (samples.table[locs1[i], phenotypes[j]] == 1) {
                              sample.phen[i] <- 1
                    }
                }
              }
         }
         print("sample.phen")
         print(sample.phen)
         
         m.1 <- m.1[, sample.phen == 1]
         sample.names.1 <- colnames(m.1)
         Ns.1 <- ncol(m.1) 
         print(paste("Matching phenotypes total number of samples:", ncol(m.1)))
         print(dim(m.1))
         print(sample.names.1)
   }

   print(dim(m.1))
   CCBA_write.gct.v1(gct.data.frame = data.frame(m.1), descs = row.names(m.1), filename = output_dataset)
}

#-------------------------------------------------------------------------------------------------	
   CCBA_match_and_merge_datasets.v1 <- function(
   #
   # Merge two datasets matching theor features or samples
   # P. Tamayo Jan 17, 2016
   #
   input_dataset1,
   input_annot_file1 = NULL,
   input_dataset2,
   input_annot_file2 = NULL,
   match_rows = T,   # T = combine samples using common rows, F = combine rows using common columns
   output_dataset,
   output_annot_file = NULL) {

# start of methodology

# Read input datasets

   dataset1 <- CCBA_read_GCT_file.v1(filename = input_dataset1)
   m1 <- data.matrix(dataset1$ds)
   gs.names1 <- dataset1$row.names
   gs.descs1 <- dataset1$descs
   sample.names1 <- dataset1$names

   dataset2 <- CCBA_read_GCT_file.v1(filename = input_dataset2)
   m2 <- data.matrix(dataset2$ds)
   gs.names2 <- dataset2$row.names
   gs.descs2 <- dataset2$descs
   sample.names2 <- dataset2$names

# Match features to first dataset and create matching m2 dataset

   if (match_rows == T) {  # combine samples using common rows

      gs.names3 <- intersect(gs.names1, gs.names2)
      print(paste("size of overlap (rows):", length(gs.names3)))
      
      locations1 <- match(gs.names3, gs.names1, nomatch=0)
      m1 <- m1[locations1, ]
      gs.descs1 <- gs.descs1[locations1]

      locations2 <- match(gs.names3, gs.names2, nomatch=0)
      m2 <- m2[locations2, ]
      gs.descs2 <- gs.descs2[locations2]

      # Merge datasets

      m3 <- cbind(m1, m2)
      sample.names3 <- c(sample.names1, sample.names2)

      # Save dataset

     V <- as.matrix(m3)
     row.names(V) <- gs.names3
     CCBA_write.gct.v1(gct.data.frame = V, descs = gs.descs1, filename = output_dataset)

  } else { # combine rows using common columns

      sample.names3 <- intersect(sample.names1, sample.names2)
      print(paste("size of overlap (columns):", length(sample.names3)))
      
      locations1 <- match(sample.names3, sample.names1, nomatch=0)
      m1 <- m1[, locations1]

      locations2 <- match(sample.names3, sample.names2, nomatch=0)
      m2 <- m2[, locations2]

      # Merge datasets

      m3 <- rbind(m1, m2)
      gs.names3 <- c(gs.names1, gs.names2)
      gs.descs3 <- c(gs.descs1, gs.descs2)      

      # Save dataset

      V <- as.matrix(m3)
      row.names(V) <- gs.names3
      CCBA_write.gct.v1(gct.data.frame = V, descs = gs.descs3, filename = output_dataset)
  }
}

#-------------------------------------------------------------------------------------------------	
    CCBA_subset_dataset_based_on_feature.v1 <- function(
    #
    # Subset dataset bassed on specific multiple features/values from a second dataset
    # P. Tamayo Jan 17, 2016
    #
    input_dataset1,
    input_dataset2,
    features,
    values,    
    output_dataset1)
   {

  # Read input datasets

   dataset1 <- CCBA_read_GCT_file.v1(filename = input_dataset1)
   m1 <- data.matrix(dataset1$ds)
   sample.names.1 <- colnames(m1)
   print(dim(m1))

   dataset2 <- CCBA_read_GCT_file.v1(filename = input_dataset2)
   m2 <- data.matrix(dataset2$ds)
   sample.names.2 <- colnames(m2)   
   print(dim(m2))
   
   # Match samples
   
   overlap <- intersect(colnames(m1), colnames(m2))
   print(paste("Size of overlap", length(overlap)))
   locs1 <- match(overlap, colnames(m1))
   locs2 <- match(overlap, colnames(m2))
   m1 <- m1[, locs1]
   sample.names.1 <- sample.names.1[locs1]
   print(dim(m1))
   m2 <- m2[, locs2]
   sample.names.2 <- sample.names.2[locs2]   
   print(dim(m2))   

  # Select subset of columns in m1 where the features have the right selection values in m2

   locs <- NULL
   for (j in 1: ncol(m1)) {
      for (k in 1:length(features)) {
#         print(paste("j:", j, " k:", k))
#         print(paste("feature:", features[k]))
#         print(paste("content:", m2[features[k], j]))                 
         if (m2[features[k], j] == values[k]) locs <- c(locs, j)
      }
   }
   locs <- unique(locs)
   print(paste("Selected:", length(locs), " from a total of ", ncol(m1), " columns"))
   m1 <- m1[, locs]
   sample.names.1 <- sample.names.1[locs]
   colnames(m1) <- sample.names.1

   print(dim(m1))
   print(colnames(m1))
   CCBA_write.gct.v1(gct.data.frame = data.frame(m1), descs = row.names(m1), filename = output_dataset1)
}

#-------------------------------------------------------------------------------------------------	
   CCBA_make_panel_of_genomic_features.v1 <- function(
   #
   # Make a multipanel heatmap of selected features (from different files)
   # P. Tamayo Jan 17, 2016
   #
      input_dataset,
      target,
      feature.files,                           
      features,
      feature.aliases = NULL,
      output.file,
      description            = "",
      sort.by.target         = T,
      rank.target            = F,
      direction              = "positive",
      missing.value.color    = "khaki1",                  # Missing feature color
      binary.0_value.color   = "lightgray",                 # Binary feature's 0's color 
      binary.1_value.color   = "black",                 # Binary feature's 1's color
      character.scaling      = 1.5,
      n.perm                 = 10000,
      create.feature.summary = F,
      feature.combination.op = "max",
      exclude.feature.NA.vals = F,
      left_margin            = 20,
      max.entries            = 32,
      pdf.size               = c(14, 11),
      nicknames              = NULL,
      target.style           = "color.bar",          # "color.bar" or "bar.graph"
      show.samples.names     = F,
      sort_features_in_panel = F,
      exclude_flat_features  = F,
      l.panels               = NULL)
  {

   set.seed(5209761)

   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   max.cont.color <- 512
   mycol <- c(mycol,
              missing.value.color,                  # Missing feature color
              binary.0_value.color,                 # Binary feature's 0's color 
              binary.1_value.color)                 # Binary feature's 1's color 

   categ.col <- c("#9DDDD6", # dusty green
                     "#F0A5AB", # dusty red
                     "#9AC7EF", # sky blue
                     "#F970F9", # violet
                     "#FFE1DC", # clay
                     "#FAF2BE", # dusty yellow
                     "#AED4ED", # steel blue
                     "#C6FA60", # green
                     "#D6A3FC", # purple
                     "#FC8962", # red
                     "#F6E370", # orange
                     "#F0F442", # yellow
                     "#F3C7F2", # pink
                     "#D9D9D9", # grey
                     "#FD9B85", # coral
                     "#7FFF00", # chartreuse
                     "#FFB90F", # goldenrod1
                     "#6E8B3D", # darkolivegreen4
                     "#8B8878", # cornsilk4
                     "#7FFFD4") # aquamarine

   binary.col <- c(binary.0_value.color,  binary.1_value.color)

   cex.size.table <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9,   # 1-10 characters
                       0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, # 11-20 characters
                       0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6)

   pdf(file=output.file, height=pdf.size[1], width=pdf.size[2])
#   tiff(filename = output.file, width = 1200, height = 1200, compression = "lzw")
#    png(filename = output.file, width = 1200, height = 1200)

   if (is.null(l.panels)) {
      n.panels <- length(feature.files) 
      l.panels <- NULL

      if (target.style == "color.bar") {          
         for (l in 1:n.panels) l.panels <- c(l.panels, 1.5, length(features[[l]]))
     } else if (target.style == "bar.graph") {                   
         for (l in 1:n.panels) l.panels <- c(l.panels, 2, length(features[[l]]))
     } else {
           stop(paste("ERROR: unknown target style", target.style))
     }
      l.panels[l.panels < 2] <- 1.5
      empty.panel <- max.entries - sum(unlist(l.panels))
      l.panels <- c(l.panels, empty.panel)
   }       
   n.panels <- length(l.panels)

   print(paste("n.panels", n.panels))
   print("l.panels")
   print(paste(l.panels, collapse=","))
      
   nf <- layout(matrix(c(seq(1, n.panels - 1), 0), n.panels, 1, byrow=T), 1, l.panels,  FALSE)
                      
   for (f in 1:length(feature.files)) {   # loop over feature types
      print(paste("Processing feature file:", feature.files[[f]]))
      dataset <- CCBA_read_GCT_file.v1(filename = input_dataset)
      m.1 <- data.matrix(dataset$ds)
      sample.names.1 <- colnames(m.1)
      Ns.1 <- ncol(m.1)
      print(paste("Total samples in input file:", Ns.1))

      target.vec <- m.1[target,]
      non.nas <- !is.na(target.vec)
      sample.names.1 <- sample.names.1[non.nas]
      m.1 <- m.1[, non.nas]
      Ns.1 <- ncol(m.1)
      
      print(paste("Total non-NAs target samples in input file:", Ns.1))
      
      dataset.2 <- CCBA_read_GCT_file.v1(filename = feature.files[[f]])
      m.2 <- data.matrix(dataset.2$ds)
      dim(m.2)
      row.names(m.2) <- dataset.2$row.names
      Ns.2 <- ncol(m.2)  
      sample.names.2 <- colnames(m.2) <- dataset.2$names

      print(paste("Total samples in features file:", Ns.2))

      if (exclude.feature.NA.vals == T) {

         feature.non.nas <- rep(1, ncol(m.2))
       
         for (feat.n in 1:length(features[[f]])) { 
           feature.name <- unlist(features[[f]][feat.n])
           for (j in 1:ncol(m.2)) {
               print(paste("feature.name:", feature.name, " j:", j, " dim m.2:", dim(m.2)))
               flush.console()
               if (is.na(m.2[feature.name, j])) feature.non.nas[j] <- 0
           }
         }
         m.2 <- m.2[, feature.non.nas == 1]
         Ns.2 <- ncol(m.2)  
         sample.names.2 <- colnames(m.2) 

         print(paste("Total non-NAs samples in feature file:", Ns.2))
     }
      
      overlap <- intersect(sample.names.1, sample.names.2)
      locs1 <- match(overlap, sample.names.1)
      locs2 <- match(overlap, sample.names.2)
      m.1 <- m.1[, locs1]
      m.2 <- m.2[, locs2]
      Ns.1 <- ncol(m.1)
      Ns.2 <- ncol(m.2)
      sample.names.1 <- colnames(m.1)
      sample.names.2 <- colnames(m.2)
      
      print(paste("feature file overlap with target samples:", length(overlap)))

      target.vec <- m.1[target,]
      if (rank.target == T) target.vec <- rank(target.vec)      

      if (sort.by.target == T) {
         ind <- order(target.vec, decreasing=T)
         target.vec <- target.vec[ind]
         sample.names.1 <- sample.names.1[ind]
         m.1 <- m.1[, ind]
         m.2 <- m.2[, ind]         
         sample.names.2 <- sample.names.2[ind]
      }
      if (direction == "negative") {
         if (length(table(target.vec)) > length(target.vec)*0.5) { # continuous target
            target.vec <- -target.vec
         } else {
            target.vec <-  1 - target.vec
         }
      } else if (direction == "positive") {

      } else {
         stop(paste("Unknown direction:", direction))
      }

    # normalize target
      unique.target.vals <- unique(target.vec)
      n.vals <- length(unique.target.vals)
      if (n.vals >= length(target.vec)*0.5) {    # Continuous value color map        
         cutoff <- 2.5
         x <- target.vec
         x <- (x - mean(x))/sd(x)         
         target.vec.norm <- x
         x[x > cutoff] <- cutoff
         x[x < - cutoff] <- - cutoff
         x <- ceiling((max.cont.color - 1) * (x + cutoff)/(cutoff*2)) + 1
         target.vec <- x
      }
  #    if (f == 1) {
  #        main <- description
  #    } else {
  #        main <- ""
  #    }
      main <- names(feature.files)[[f]]
#      par(mar = c(0, 16, 2, 12))
      par(mar = c(0, left_margin, 4, 12))

      target.nchar <- ifelse(nchar(target) > 30, 30, nchar(target))
#      cex.axis <- cex.size.table[target.nchar]
      cex.axis <- 1
#      print(paste("cex.axis:", cex.axis))      
      
      if (n.vals >= length(target.vec)*0.5) {    # Continuous value color map

       if (target.style == "color.bar") {          
          image(1:Ns.1, 1:1, as.matrix(target.vec), zlim = c(0, max.cont.color), col=mycol[1: max.cont.color],
                axes=FALSE, main=main, sub = "", xlab= "", ylab="", cex.main=1.35*character.scaling)
          ref1 <- 1
          ref2 <- 1
          
       } else if (target.style == "bar.graph") {
            V1.vec <- as.vector(target.vec.norm)
            V1.vec <- (V1.vec - min(V1.vec))/(max(V1.vec) - min(V1.vec))
            barplot(V1.vec, xaxs="i", ylim=c(-0.25, 1), col="darkgrey", border="darkgrey", xaxt='n', yaxt='n',
                    ann=FALSE, bty='n', width=1, space=0)
           points(seq(1, length(V1.vec)), V1.vec, type="l", col=1)
           points(rep(0, 11), seq(0, 1, 0.1), type="l", col=1)
           points(seq(1, length(V1.vec)), rep(0, length(V1.vec)), type="l", col=1)
           ref1 <- 0.5
            ref2 <- -0.25
           
       } else {
           stop(paste("ERROR: unknown target style", target.style))
       }


      } else if (n.vals == 2) {  # binary
         image(1:Ns.1, 1:1, as.matrix(target.vec), zlim = range(target.vec), col=binary.col, axes=FALSE, cex.main=1.35*character.scaling, main=main, sub = "", xlab= "", ylab="",
                cex.main=2)
          ref1 <- 1
          ref2 <- 1
          target.vec.norm <- target.vec         
   
      } else {  # categorical
         image(1:Ns.1, 1:1, as.matrix(target.vec), zlim = range(target.vec), col=categ.col[1:n.vals], cex.main=1.35*character.scaling, axes=FALSE, main=main, sub = "",
               xlab= "", ylab="", cex.main=2)
          ref1 <- 1
          ref2 <- 1
          target.vec.norm <- target.vec                  

      }

      if (!is.na(match(target, names(nicknames)))) {
          target.name <- nicknames[target]
      } else {
           target.name <- target
      }
      axis(2, at=ref1, labels=target.name, adj= 0.5, tick=FALSE, las = 1, cex=1, cex.axis=cex.axis*character.scaling,
            line=0, font=2, family="",font.axis=1)
      axis(4, at=ref2, labels=paste(" IC     p-value"), adj= 0.5, tick=FALSE, las = 1, cex=1, cex.axis=cex.axis*character.scaling,
           font.axis=1, line=0, font=2, family="")

      feature.mat <- feature.names <- NULL
      
     for (feat.n in 1:length(features[[f]])) { 
        len <- length(unlist(features[[f]][feat.n]))         
        feature.name <- unlist(features[[f]][feat.n])
        print(paste("      Feature:", feature.name))
        if (is.na(match(feature.name, row.names(m.2)))){
        print(paste(feature.name, ' NOT FOUND !!!!!!!!!!\n\n', sep = ''))    
        next
        }

        if (exclude_flat_features == T & sum(m.2[feature.name,]) == 0) next
        
        feature.mat <- rbind(feature.mat,  m.2[feature.name,])
        if (!is.null(feature.aliases[1])) {
           if (!is.na(match(feature.name, names(feature.aliases)))) {
               feature.name <- unlist(feature.aliases[feature.name])
           }
       }
        if (!is.na(match(feature.name, names(nicknames)))) {
           feature.name <- nicknames[feature.name]
        } 
        feature.names <- c(feature.names, feature.name)
      }
      feature.mat <- as.matrix(feature.mat)
      row.names(feature.mat) <- feature.names

      if (create.feature.summary == T) {
         summary.feature <- apply(feature.mat, MARGIN=2, FUN=feature.combination.op)
         feature.mat <- rbind(feature.mat, summary.feature)
         row.names(feature.mat) <- c(feature.names, "SUMMARY FEATURE")
     }

            # compute IC association with target

      IC.vec <- p.val.vec <- stats.vec <- NULL
      
      for (i in 1:nrow(feature.mat)) {
           feature.vec <- feature.mat[i,]
           IC <- CCBA_IC.v1(feature.vec, target.vec.norm)
#           print(paste("Feature:", row.names(feature.mat)[i], " IC=", IC))
           IC <- signif(IC, 3)
           null.IC <- vector(length=n.perm, mode="numeric")
           for (h in 1:n.perm) null.IC[h] <- CCBA_IC.v1(feature.vec, sample(target.vec))
           if (IC >= 0) {
             p.val <- sum(null.IC >= IC)/n.perm
           } else {
             p.val <- sum(null.IC <= IC)/n.perm
           }
           p.val <- signif(p.val, 3)
           if (p.val == 0) {
#             p.val <- paste("< ", signif(1/n.perm, 3), sep="")
              p.val <- signif(1/n.perm, 3)
           }
           IC.vec <- c(IC.vec, IC)
           p.val.vec <- c(p.val.vec, p.val)
           space.chars <- "           "
           IC.char <- nchar(IC)
           pad.char <- substr(space.chars, 1, 10 - IC.char)
           stats.vec <- c(stats.vec, paste(IC, pad.char, p.val, sep=""))
       }

     for (i in 1:nrow(feature.mat)) {

           feature.vec <- feature.mat[i,]
           unique.feature.vals <- unique(sort(feature.vec))
           non.NA.vals <- sum(!is.na(feature.vec))
           n.vals <- length(unique.feature.vals)
           if (n.vals > 2) {    # Continuous value color map        
              feature.vals.type <- "continuous"
              cutoff <- 2.5
              x <- feature.vec
              locs.non.na <- !is.na(x)
              x.nonzero <- x[locs.non.na]
              x.nonzero <- (x.nonzero - mean(x.nonzero))/sd(x.nonzero)         
              x.nonzero[x.nonzero > cutoff] <- cutoff
              x.nonzero[x.nonzero < - cutoff] <- - cutoff      
              feature.vec[locs.non.na] <- x.nonzero
              feature.vec2 <- feature.vec
              feature.vec[locs.non.na] <- ceiling((max.cont.color - 2) * (feature.vec[locs.non.na] + cutoff)/(cutoff*2)) + 1
              feature.vec[is.na(x)] <- max.cont.color + 1
              feature.mat[i,] <- feature.vec              
           }
      }
      feature.mat <- as.matrix(feature.mat)

      if (sort_features_in_panel == T & nrow(feature.mat) > 1) {
          ind <- order(IC.vec, decreasing=T)
          IV.vec <- IC.vec[ind]
          stats.vec <- stats.vec[ind]
          feature.mat <- feature.mat[ind,]
      }
      
      print("feature matrix dimensions:")
      print(paste(nrow(feature.mat), ncol(feature.mat)))
      
      if (nrow(feature.mat) > 1) {
          V <- apply(feature.mat, MARGIN=2, FUN=rev)
      } else {
          V <- as.matrix(feature.mat)
      }
      
      features.max.nchar <- max(nchar(row.names(V)))
      features.nchar <- ifelse(features.max.nchar > 30, 30, features.max.nchar)
#      cex.axis <- cex.size.table[features.nchar]
#      print(paste("cex.axis:", cex.axis))

#      par(mar = c(1, 2, 1, 12))
      par(mar = c(3, left_margin, 1, 12))
#      par(mar = c(1.5, left_margin, 1, 12))            

      main <- names(feature.files)[[f]]
      
     if (n.vals > 2) {    # Continuous value color map        
         image(1:dim(V)[2], 1:dim(V)[1], t(V), zlim = c(0, max.cont.color + 3), col=mycol, axes=FALSE, main="", cex.main=0.8, sub = "", xlab= "", ylab="")
     } else {  # binary
         image(1:dim(V)[2], 1:dim(V)[1], t(V), zlim = c(0, 1), col=binary.col, axes=FALSE, main="", cex.main=0.4,  sub = "", xlab= "", ylab="")
     }
     axis(2, at=1:dim(V)[1], labels=row.names(V), adj= 0.5, tick=FALSE, las = 1, cex=1, cex.axis=cex.axis*character.scaling, font.axis=1, line=0, font=2, family="")
     axis(4, at=1:dim(V)[1], labels=rev(stats.vec), adj= 0.5, tick=FALSE, las = 1, cex=1, cex.axis=cex.axis*character.scaling, font.axis=1, line=0, font=2, family="")
      if (show.samples.names == T) {
           axis(1, at=1:dim(V)[2], labels=colnames(V), adj= 0.5, tick=FALSE, las = 3, cex=1, cex.axis=0.5*cex.axis*character.scaling, font.axis=1, line=-1, font=2, family="")
           col.names <- colnames(V)
           for (k in 1:length(col.names)) print(paste(k, col.names[k]))
      }

 
     }
   dev.off()
}

#-------------------------------------------------------------------------------------------------	
    CCBA_gene_sets_overlap.v1 <- function(
    #
    # Compute the overlap between two gene sets (GMT files)
    # P. Tamayo Jan 17, 2016
    #
    gmt.file1,
    gmt.file2,
    gene.set1,
    gene.set2,
    gmt.name,
    gmt.output)

    {
       if (gmt.file1 == gmt.file2) {
          gene.set.files <- gmt.file1
       } else {
          gene.set.files <- c(gmt.file1, gmt.file2)
       }
        
	# Read gene set files
	
	max.G <- 0
	max.N <- 0
	for (gsdb in gene.set.files) {
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		max.G <- max(max.G, max(GSDB$size.G))
		max.N <- max.N +  GSDB$N.gs
	}
	N.gs <- 0
	gs <- matrix("null", nrow=max.N, ncol=max.G)
	gs.names <- vector(length=max.N, mode="character")
	gs.descs <- vector(length=max.N, mode="character")
	size.G <- vector(length=max.N, mode="numeric")
	start <- 1
	for (gsdb in gene.set.files) {
		GSDB <- CCBA_Read.GeneSets.db.v1(gsdb, thres.min = 2, thres.max = 2000, gene.names = NULL)
		N.gs <- GSDB$N.gs 
		gs.names[start:(start + N.gs - 1)] <- GSDB$gs.names
		gs.descs[start:(start + N.gs - 1)] <- GSDB$gs.desc
		size.G[start:(start + N.gs - 1)] <- GSDB$size.G
		gs[start:(start + N.gs - 1), 1:max(GSDB$size.G)] <- GSDB$gs[1:N.gs, 1:max(GSDB$size.G)]
		start <- start + N.gs
	}
	N.gs <- max.N

        loc1 <- match(gene.set1, gs.names)
        set1 <- gs[loc1,]
        set1 <- gs[loc1, 1:size.G[loc1]]
        loc2 <- match(gene.set2, gs.names)
        set2 <- gs[loc2,]
        set2 <- gs[loc2, 1:size.G[loc2]]

        print(paste("Size of gene set 1", gene.set1, " = ", length(set1)))
        print(paste("Size of gene set 2", gene.set2, " = ", length(set2)))       
        overlap <- intersect(set1, set2)
        print(paste("Size of overlap = ", length(overlap)))
        print("Overlap:")
        print(overlap)                     

        row.header <- gmt.name
        output.line <- paste(overlap, collapse="\t")
        output.line <- paste(row.header, row.header, output.line, sep="\t")
        write(noquote(output.line), file = gmt.output, append = F, ncolumns = length(overlap) + 2)

   }

#-------------------------------------------------------------------------------------------------	
   CCBA_IC_selection_features_overlap.v1 <- function(
   #
   # Compute overlap in the top features from IC selection analysis
   # P. Tamayo Jan 17, 2016
   #
   file1,
   file2,
   direction_file2   = "positive", # "negative"
   exclude_suffix1   = F,
   exclude_suffix2   = F,    
   n.markers         = 50,
   output.file,
   gmt.names         = NULL,
   gmt.file          = NULL,
   append.gmt        = F)
   {

   write("Overlap Analysis", file = output.file, ncolumns = 50, append = FALSE, sep = "")
   write(file1, file = output.file, ncolumns = nchar(file1), append  =TRUE, sep = "\t")
   write(file2, file =output.file, ncolumns = nchar(file2), append =TRUE, sep = "\t")

   df1 <- read.table(file1, header=T, row.names=1, sep="\t", skip=0)
   df2 <- read.table(file2, header=T, row.names=1, sep="\t", skip=0)

  features1 <- df1[,"Feature"]
  features2 <- df2[,"Feature"]

  if (direction_file2 == "negative") features2 <- rev(features2)

  up.markers1.orig <- up.markers1 <- as.character(features1[1:n.markers])
  dn.markers1.orig <- dn.markers1 <- as.character(features1[seq(length(features1) - n.markers + 1, length(features1))])

  for (i in 1:length(up.markers1)) up.markers1[i] <- strsplit(up.markers1[i], " ")[[1]]
  for (i in 1:length(dn.markers1)) dn.markers1[i] <- strsplit(dn.markers1[i], " ")[[1]]   
   
  if (exclude_suffix1 == T) {
     for (i in 1:length(up.markers1)) up.markers1[i] <- strsplit(up.markers1[i], "_")[[1]]
     for (i in 1:length(dn.markers1)) dn.markers1[i] <- strsplit(dn.markers1[i], "_")[[1]]
  }

  up.markers2.orig <- up.markers2 <- as.character(features2[1:n.markers])
  dn.markers2.orig <- dn.markers2 <- as.character(features2[seq(length(features2) - n.markers + 1,  length(features2))])

  for (i in 1:length(up.markers2)) up.markers2[i] <- strsplit(up.markers2[i], " ")[[1]]
  for (i in 1:length(dn.markers2)) dn.markers2[i] <- strsplit(dn.markers2[i], " ")[[1]]   
   
  if (exclude_suffix2 == T) {
     for (i in 1:length(up.markers2)) up.markers2[i] <- strsplit(up.markers2[i], "_")[[1]]
     for (i in 1:length(dn.markers2)) dn.markers2[i] <- strsplit(dn.markers2[i], "_")[[1]]
  }
   
  print("                 ")
  write("          ", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print("Up Markers Overlap")
  write("Up Markers Overlap", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  up.overlap <- intersect(up.markers1, up.markers2)
  print(paste("Size of up overlap:", length(up.overlap)))
  write(paste("Size of up overlap:", length(up.overlap)), file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print(cbind(up.overlap))
  write.table(cbind(up.overlap), file=output.file, quote=F, col.names = T, row.names = F, append = T, sep="\t")

  if (!is.null(gmt.names)) {
      row.header <- gmt.names[1]
      output.line <- paste(up.overlap, collapse="\t")
      output.line <- paste(row.header, row.header, output.line, sep="\t")
      write(noquote(output.line), file = gmt.file, append = append.gmt, ncolumns = length(up.overlap) + 2)
  }
   
  print("                 ")
  write("          ", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print("Down Markers Overlap")
  write("Down Markers Overlap", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  dn.overlap <- intersect(dn.markers1, dn.markers2)
  print(paste("Size of down overlap:", length(dn.overlap)))
  write(paste("Size of down overlap:", length(dn.overlap)), file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print(cbind(dn.overlap))
  write.table(cbind(dn.overlap), file=output.file, quote=F, col.names = T, row.names = F, append = T, sep="\t")

  if (!is.null(gmt.names)) {
      row.header <- gmt.names[2]
      output.line <- paste(dn.overlap, collapse="\t")
      output.line <- paste(row.header, row.header, output.line, sep="\t")
      write(noquote(output.line), file = gmt.file, append = T, ncolumns = length(dn.overlap) + 2)
  }

  print("                 ")
  write("          ", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print("Up Marker Features")
  write("Up Markers Features", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  locs <- match(up.markers1.orig, features2)

  up.table <- cbind(as.character(up.markers1.orig), seq(1, n.markers),locs, ifelse(locs <= rep(n.markers, n.markers), rep("*", n.markers),rep(" ", n.markers)))
  colnames(up.table) <- c("Feature", "Rank in file1", "Rank in file2", "Overlap")
  print(noquote(up.table))
  write.table(cbind(up.table), file=output.file, quote=F, col.names = T, row.names = F, append = T, sep="\t")

  print("                 ")
  write("          ", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  print("Down Marker Features")
  write("Down Markers Features", file =output.file, ncolumns = 50, append =TRUE, sep = "\t")
  locs <- match(dn.markers1.orig, features2)
  locs <- length(features2) - locs + 1
  dn.table <- cbind(as.character(dn.markers1.orig), seq(1, n.markers), locs, ifelse(locs <= rep(n.markers, n.markers), rep("*", n.markers),  rep(" ", n.markers)))
  colnames(dn.table) <- c("Feature", "Rank in file1", "Rank in file2", "Overlap")
  print(noquote(dn.table))
  write.table(cbind(dn.table), file=output.file, quote=F, col.names = T, row.names = F, append = T, sep="\t")

}

#-------------------------------------------------------------------------------------------------	
   CCBA_OncoGPS_project_dataset.v1 <- function(
   # 
   # Project a dataset in the space defined by a W matrix
   # P. Tamayo Jan 17, 2016
   #    
      input_dataset,                    # Input dataset (GCT)
      input_normalization = "rank",     # Normalization for the input dataset: "rank"
      normalize_after_match = T,        # Normalize input dataset after matching with rows of W
      projection_method = "NNLS",       # Projection Method: NNLS=Non Negative Linear Solver, INV= Pseudo Inverse  
      input_W_dataset,                  # Input W matrix (GCT)
      W_normalization = "none",         # Normalization for W                                            
      output_H_dataset,                 # Output dataset H (GCT)
      output_W_dataset = NULL)          # Output dataset normalized W (GCT)                                            
  {

   set.seed(5209761)
       
   # Read input dataset

   dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   m <- data.matrix(dataset.1$ds)
   print(dim(m))

   if (normalize_after_match == F) {  # Normalize input data here before matching with W
     if (input_normalization == "rank") {
         print("A before normalization:")

         max.n <- 10000
         for (i in 1:ncol(m)) m[,i] <- (max.n - 1) * (rank(m[,i]) - 1) /(nrow(m) - 1) + 1

         print("A after normalization:")
   
      } else if (input_normalization == "none") {   

      } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
      }
   }
   
   dataset.2 <- CCBA_read_GCT_file.v1(filename = input_W_dataset)
   W <- data.matrix(dataset.2$ds)
   print(dim(W))

   k.comp <- ncol(W)

  # match gene lists

  overlap <- intersect(row.names(W), row.names(m))
  print(paste("overlap:", length(overlap)))
  locs1 <- match(overlap, row.names(m))
  locs2 <- match(overlap, row.names(W))
  m <- m[locs1,]
  W <- W[locs2,]

  ind <- order(row.names(W))
  W <- W[ind,]
  m <- m[ind,]

  if (normalize_after_match == T) {  # Normalize input data here after matching with W
     if (input_normalization == "rank") {

         max.n <- 10000
         for (i in 1:ncol(m)) m[,i] <- (max.n - 1) * (rank(m[,i]) - 1) /(nrow(m) - 1) + 1
 
      } else if (input_normalization == "none") {   

      } else {
         stop(paste("ERROR: unknown input normalization:", input_normalization))  
      }
   }

  # Normalize W

  if (W_normalization == "equal_sum") {
     norm.factors <- apply(W, MARGIN=2, FUN=sum)
     for (i in 1:ncol(W)) {
        W[, i] <- 1000*W[, i]/norm.factors[i]
      }
   }

   H <- matrix(0, nrow=k.comp, ncol= ncol(m), dimnames=list(colnames(W), colnames(m)))
   if (projection_method == "NNLS") {   # non-negative linear solver
      for (i in 1:ncol(H)) H[, i] <- nnls.fit(W, m[, i], wsqrt=1, eps=0, rank.tol=1e-07)
   } else if (projection_method == "INV") {  # Pseudo-inverse
       H <- ginv(W) %*% m
       row.names(H) <- colnames(W)
       colnames(H) <- colnames(m)
       H[H < 0] <- .Machine$double.eps   # Eliminate negative entries
   } else {
       stop(paste("ERROR: Unknown projection method:", projection_method))
   }

   # Save H matrix

   CCBA_write.gct.v1(gct.data.frame = H, descs = row.names(H), filename = output_H_dataset)

   # Save (optional) normalized W
   
   if(!(is.null(output_W_dataset))) CCBA_write.gct.v1(gct.data.frame = W, descs = row.names(W), filename = output_W_dataset)

 }

#-------------------------------------------------------------------------------------------------	
   CCBA_OncoGPS_populate_map.v1 <- function(
   # 
   # Display feature(s) in OncoGPS map
   # P. Tamayo Jan 17, 2016
   #    
      projection_dataset,
      projection.set.norm     = "wrt_train", # "wrt_train", "wrt_test" 
      feature.files,
      features,
      produce.boxplots        = T,
      n.perm                  = 100,
      output.file             = NULL,
      OPM.objects.file,
      point.cex               = 2,
      plot_sample_names       = F,
      cex_sample_names        = 1,       
      show_missing_samples    = T)
   {

      new.point.cex <- point.cex
      load(OPM.objects.file)
      point.cex <- new.point.cex
      
      if (!is.null(output.file)) pdf(file=output.file, height=8.5, width=11)
      
      dataset.1.projection <- CCBA_read_GCT_file.v1(filename = projection_dataset)
      m.1.projection <- data.matrix(dataset.1.projection$ds)
      dim(m.1.projection)
      Ns.1.projection <- ncol(m.1.projection)
      print(paste("Total samples in projection file:", Ns.1.projection))
      sample.names.1.projection <- colnames(m.1.projection) <- dataset.1.projection$names

      # Check if all the nodes can be mapped

      locs <- match(nodes, row.names(m.1.projection))
      H.projection <- m.1.projection[locs,]
   
      print("Performing normalization")
      
      if (projection.set.norm == "wrt_test") {
         nodes.mean <- apply(H.projection, MARGIN=1, FUN=mean)
         nodes.sd   <- apply(H.projection, MARGIN=1, FUN=sd)
         for (i in 1:nrow(H.projection)) {
            H.projection[i,] <- (H.projection[i,] - nodes.mean[i])/nodes.sd[i]
            for (j in 1:ncol(H.projection)) {
               if (H.projection[i, j] >  norm.thres) H.projection[i, j] <- norm.thres
               if (H.projection[i, j] < -norm.thres) H.projection[i, j] <- -norm.thres
            }
         }
         nodes.min <- apply(H.projection, MARGIN=1, FUN=min)
         nodes.max <- apply(H.projection, MARGIN=1, FUN=max)
         for (i in 1:nrow(H.projection)) {   
            H.projection[i,] <- (H.projection[i,] - nodes.min[i])/(nodes.max[i] - nodes.min[i])           
         }
      } else if (projection.set.norm == "wrt_train") {
         for (i in 1:nrow(H.projection)) {
            H.projection[i,] <- (H.projection[i,] - nodes.mean[i])/nodes.sd[i]
            for (j in 1:ncol(H.projection)) {
               if (H.projection[i, j] >  norm.thres) H.projection[i, j] <- norm.thres
               if (H.projection[i, j] < -norm.thres) H.projection[i, j] <- -norm.thres
            }
         }
         for (i in 1:nrow(H.projection)) {   
            H.projection[i,] <- (H.projection[i,] - nodes.min[i])/(nodes.max[i] - nodes.min[i])           
         }
      }
      
      print(dim(H.projection))
      print("H.projecting matrix after norm, in test mode:")
      flush.console()      
      print(H.projection[, 1:3])

      predicted.state.test <- predict(svm.mod, newdata = t(H.projection))
      print("predicted state test")
      print(predicted.state.test[1:10])
      print(table(predicted.state.test))
      flush.console()   
      
      # Compute triangle membership and xp, yp 2D coordinates

      tri.weig.projection <- matrix(0, nrow=ncol(H.projection), ncol=length(triangles))
      tri.member.projection <- rep(0, ncol(H.projection))

      # Assign samples to triangles

      for (i in 1:ncol(H.projection)) {
         for (tri in 1:length(triangles)) {
            tri.weig.projection[i, tri] <- sum(H.projection[triangles[[tri]], i])
         }
         tri.member.projection[i] <- which.max(tri.weig.projection[i,])
      }
      xp.projection <- yp.projection <- xp.local.projection <- yp.local.projection <- rep(0, ncol(H.projection))

      P <- c(0, 0)
      Q <- c(1, 0)
      R <- c(1/2, sqrt(3)/2)
      Tran <- rbind(c(P - R, 0), c(Q - R, 0), c(R, 1))

      H.prime <- H.projection

      # only keep 3 top stronger nodes

      H.prime2 <- matrix(0, nrow=nrow(H.prime), ncol=ncol(H.prime))
      for (j in 1:ncol(H.prime2)) {
         top.vals <- order(H.prime[,j], decreasing=T)[1:3]
         H.prime2[top.vals,j] <- H.prime[top.vals,j]
      }
      
      for (j in 1:ncol(H.prime2)) {
         weight <- sum(H.prime2[,j]^expon)
         xp.projection[j] <- sum(H.prime2[,j]^expon*row.objects[,1])/weight
         yp.projection[j] <- sum(H.prime2[,j]^expon*row.objects[,2])/weight                      
       }

     for (f in 1:length(feature.files)) {   # loop over feature types
        print(paste("Processing feature file:", feature.files[[f]][1]))
        dataset.2 <- CCBA_read_GCT_file.v1(filename = feature.files[[f]][1])
        m.2 <- data.matrix(dataset.2$ds)
        dim(m.2)
        row.names(m.2) <- dataset.2$row.names
        sample.names.2 <- colnames(m.2) <- dataset.2$names
        if (feature.files[[f]][2] != "ALL") {
           overlap <- intersect(sample.names.2, feature.files[[f]][2:length(feature.files[[f]])])
           locs0 <- match(overlap, sample.names.2)
           m.2 <- m.2[, locs0]
        }
        sample.names.2 <- colnames(m.2) 
        Ns.2 <- ncol(m.2)  
        locs <- match(sample.names.1.projection, sample.names.2)
        locs1 <- seq(1, length(locs))[!is.na(locs)]
        locs2 <- locs[!is.na(locs)]
        m <- matrix(NA, nrow=nrow(m.2), ncol=Ns.1.projection, dimnames=list(row.names(m.2), sample.names.1.projection))
        m[, locs1] <- m.2[, locs2]
        m.2 <- m

        print(paste("feature file overlap with projection dataset:", ncol(m.2)))

        if (length(features[[f]]) == 1 & is.numeric(features[[f]])) {    # search for top matching features from that feature file 
           print(paste("Search for top features for feature file:", feature.files[[f]][1]))
           print(paste("samples:", feature.files[[f]][2]))         
           feat.IC <- rep(0, nrow(m.2))
           for (h in 1:nrow(m.2)) feat.IC[h] <- abs(CCBA_IC.v1(m.2[h,], match(predicted.state.test, all.classes)))
           ind <- order(feat.IC, decreasing=T)
           feat.IC <- feat.IC[ind]
           feat.IC.names <- row.names(m.2)[ind]
           features[[f]] <- feat.IC.names[1:features[[f]]]
           print("Top matching features:")
           print(features[[f]])
        }
        flush.console()
      
        if (length(features[[f]]) == 1 & features[[f]] == "ALL") {    # display all features from that feature file 
            features[[f]] <- row.names(m.2)
        }

       # Plot projection dataset samples assigned to states

      par(mar=c(2,3,1,3))
      plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min , 1.1*y.max), bty="n", axes=F, xlab="", ylab="")               

      for (i in 1:size.grid) {
          for (j in 1:size.grid) {
              x.p <- x.coor[i]
              y.p <- y.coor[j]
              col <- mycol.class[ceiling((myncolors - 1) * final.Pxy[i, j] + 1), match(winning.class[i, j], all.classes)]
              points(x.p, y.p, col=col, pch=15, cex=1)
          }
      }
      levels <- seq(0, 1, 1/contour.levels)
      lc <- contourLines(x=x.coor, y=y.coor, z=final.Pxy, levels=levels)
      for (i in 1:length(lc)) points(lc[[i]]$x, lc[[i]]$y, type="l", col= brewer.pal(9, "Blues")[contour.tone], lwd=1)      
      
      for (i in 1:size.grid) {
          for (j in 1:size.grid) {
              x.p <- x.coor[i]
              y.p <- y.coor[j]
              if (mask[i,j] == 0) {
                 points(x.p, y.p, col="white", pch=15, cex=1)
              }
           }
       }
      
      text(x.min + 0.05*x.len, 1.08*y.max, "OncoGenic Positional System (Onco-GPS) Map",
              cex= 1.35, family="Times", pos=4, font=4, col="darkblue") # fontface="italic", 
       text(x.min + 0.05*x.len, 1.02*y.max, paste("Basic Layout: Samples (", length(xp),
              ") and States (", k.classes, ")", sep=""), cex = 1.1, font=2, family="", pos=4, col="darkred")
       text(x.min, y.min, description, cex = 1, font=2, family="", pos=4, col="darkblue")

       for (tri in 1:length(triangles)) {
           triangle.nodes.x  <- c(row.objects[triangles[[tri]][1], 1], row.objects[triangles[[tri]][2], 1], row.objects[triangles[[tri]][3], 1],
                               row.objects[triangles[[tri]][1], 1])
           triangle.nodes.y  <- c(row.objects[triangles[[tri]][1], 2], row.objects[triangles[[tri]][2], 2], row.objects[triangles[[tri]][3], 2],
                                  row.objects[triangles[[tri]][1], 2])
           points(triangle.nodes.x, triangle.nodes.y, type="l", col="black", lwd=1, cex=1.25)              
       }
   
       for (i in 1:length(xp.projection)) {
           col <- categ.col2[match(predicted.state.test[i], all.classes)]
           points(xp.projection[i], yp.projection[i], col="black", bg=col, pch=21, cex=point.cex)
       }

       for (i in 1:nrow(row.objects)) {
          pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
          text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
      }
      points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)

      leg.txt <- all.classes
      pch.vec <- rep(21, length(leg.txt))
      col <- unique(categ.col2[match(cutree.model, all.classes)])        
                        
      legend(x.max - 0.05*x.len, y.max, legend=leg.txt, bty="n", xjust=0, yjust= 1, pch = pch.vec, title="States",   
          pt.bg = col, col = "black", cex = 1, pt.cex = 1.5)

      if (plot_sample_names == T) pointLabel(xp.projection, yp.projection, labels=colnames(H.projection), cex=cex_sample_names, col="darkgreen")

      # Loop over features
        
        for (feat.n in 1:length(features[[f]])) { 

           len <- length(unlist(features[[f]][feat.n]))         
           if (len > 1) {   # is this a combined feature?
              comb.feature <- T
              f.set <- unlist(features[[f]][feat.n])
              f.type <- f.set[length(f.set)]
              f.set <- f.set[seq(1, length(f.set) - 1)]
              print(paste("combined feature:", f.set))
              print(paste("combined type:", f.type))
              feature.name <- paste(f.set, collapse=" ")
              print(paste("feature.name:", feature.name))
              if (f.type == "comb.binary") {
                 feature.vec <- apply(m.2[f.set,], MARGIN=2, FUN=max)
              } else {     # comb.categ
                 feature.vec0 <- apply(m.2[f.set,], MARGIN=2, FUN=paste, collapse="")
                 print("feature.vec0:")
                 print(feature.vec0[1:10])               

                 feature.vec <- strtoi(feature.vec0, base = 2L)
                 print("feature.vec:")
                 print(feature.vec)
                 ind <- order(feature.vec, decreasing=F)
                 u.vals <- unique(feature.vec[ind])
                 feature.vec <- match(feature.vec, u.vals) - 1
                 print("feature.vec (final):")
                 print(feature.vec)

                 f.names <- rep("other", ncol(m.2))
                 for (h in 1:ncol(m.2)) {
                    for (l in 1:length(f.set)) {
                        if (m.2[f.set[l], h] == 1) f.names[h] <- f.set[l]
                    }
                 }
                 f.names <- f.names[ind]
                 f.names <- unique(f.names)
                 print("f.names:")
                 print(f.names)
              }
           } else {
              comb.feature <- F
              f.type <- "single"                         
              feature.name <- unlist(features[[f]][feat.n])
              print(paste("      Feature:", feature.name))
              if (is.na(match(feature.name, row.names(m.2)))) {
                  print(paste("Feature not found:", feature.name))
                  next
              }
              feature.vec <- m.2[feature.name,]
              f.names <- unique(feature.vec)
           }
           flush.console()
        
           # Add feature to all features array

           #print("adding features... to all.features array")
           #print(dim(all.features))
           #print(length(feature.vec))
        
           #all.features <- rbind(all.features, feature.vec)
          # all.features.names <- c(all.features.names, feature.name)
          # all.features.descs <- c(all.features.descs, names(features)[[f]])        
           unique.feature.vals <- unique(sort(feature.vec))
           unique.feature.vals <-  unique.feature.vals[!is.na(unique.feature.vals)]
           non.NA.vals <- sum(!is.na(feature.vec))
           n.vals <- length(unique.feature.vals)
           if (n.vals >= 1*non.NA.vals/2) {    # Continuous value color map
              feature.vals.type <- "continuous"
              cutoff <- 2
              x <- feature.vec
              locs.non.na <- !is.na(x)
              x.nonzero <- x[locs.non.na]
              x.nonzero <- (x.nonzero - mean(x.nonzero))/sd(x.nonzero)         
              x.nonzero[x.nonzero > cutoff] <- cutoff
              x.nonzero[x.nonzero < - cutoff] <- - cutoff      
              feature.vec[locs.non.na] <- x.nonzero
              feature.vec2 <- feature.vec
              feature.vec[locs.non.na] <- ceiling((max.cont.color - 2) * (feature.vec[locs.non.na] + cutoff)/(cutoff*2)) + 1
              feature.vec[is.na(x)] <- max.cont.color
              col.points <- mycol[feature.vec]
           } else if (n.vals > 2 & n.vals < non.NA.vals/2 | (comb.feature == T) & (f.type == "comb.categ")) {  
              feature.vec2 <- feature.vec            
              feature.vec[is.na(feature.vec)] <- max.cont.color
              feature.vals.type <- "categorical"             
              col.points <- categ.col2[match(feature.vec, unique.feature.vals)]
           } else if (n.vals == 2 | (comb.feature == T) & (f.type == "comb.binary")) {   # Categorical color map
              feature.vec2 <- feature.vec            
              feature.vec[is.na(feature.vec)] <- max.cont.color
              feature.vals.type <- "binary"
              col.points <- binary.col[match(feature.vec, unique.feature.vals)]
           } else if (n.vals == 1) {
              feature.vec2 <- feature.vec            
              feature.vec[is.na(feature.vec)] <- max.cont.color
              feature.vals.type <- "binary"
              if (unique.feature.vals == 1) col.points <- binary.col[match(feature.vec, unique.feature.vals) + 1]
              else col.points <- binary.col[match(feature.vec, unique.feature.vals)]
           } else {
              feature.vec2 <- feature.vec            
              feature.vec[is.na(feature.vec)] <- max.cont.color
              feature.vals.type <- "binary"
              col.points <- binary.col[match(feature.vec, unique.feature.vals)]
           }

           num.classes <- match(predicted.state.test, all.classes)        
           IC <- CCBA_IC.v1(feature.vec, num.classes)
           null.IC <- vector(length=n.perm, mode="numeric")
           for (h in 1:n.perm) null.IC[h] <- CCBA_IC.v1(feature.vec, sample(num.classes))
           if (IC >= 0) {
              p.val <- sum(null.IC >= IC)/n.perm
           } else {
             p.val <- sum(null.IC <= IC)/n.perm
           }
           if (p.val == 0) {
               p.val <- paste("<", signif(1/n.perm, 3))
           } else {
               p.val <- signif(p.val, 3)
           }

           # Make plot

           par(mar=c(2,3,1,3))        
           plot(c(0,0), c(0,0), type="n", xlim=c(x.min, x.max), ylim=c(y.min - 0.065*y.len, 1.12*y.max),
                bty="n", axes=FALSE, xlab="", ylab="")                 
         for (i in 1:size.grid) {
            for (j in 1:size.grid) {
               x.p <- x.coor[i]
               y.p <- y.coor[j]
               col <- mycol.class[ceiling((myncolors - 1) * final.Pxy[i, j] + 1), match(winning.class[i, j], all.classes)]
               points(x.p, y.p, col=col, pch=15, cex=1)
            }
         }
         levels <- seq(0, 1, 1/contour.levels)
         lc <- contourLines(x=x.coor, y=y.coor, z=final.Pxy, levels=levels)
         for (i in 1:length(lc)) points(lc[[i]]$x, lc[[i]]$y, type="l", col= brewer.pal(9, "Blues")[contour.tone], lwd=1)                 

         for (i in 1:size.grid) {
            for (j in 1:size.grid) {
               x.p <- x.coor[i]
               y.p <- y.coor[j]
               if (mask[i,j] == 0) {
                  points(x.p, y.p, col="white", pch=15, cex=1)
               }
            }
          }
          for (tri in 1:length(triangles)) {
             triangle.nodes.x  <- c(row.objects[triangles[[tri]][1], 1], row.objects[triangles[[tri]][2], 1],
                                    row.objects[triangles[[tri]][3], 1], row.objects[triangles[[tri]][1], 1])
             triangle.nodes.y  <- c(row.objects[triangles[[tri]][1], 2], row.objects[triangles[[tri]][2], 2],
                                    row.objects[triangles[[tri]][3], 2], row.objects[triangles[[tri]][1], 2])
             points(triangle.nodes.x, triangle.nodes.y, type="l", col="black", lwd=1, cex=1.25)              
          }


         # Onco-GPS Map

          feature.name.short <- substr(feature.name, 1, 23)
          if (nchar(feature.name) > 25) feature.name.short <- paste(feature.name.short, "...", sep="")
          text(x.min, 1.1*y.max + 0.03*y.len, paste(feature.name.short, " (",  names(features)[[f]], ")"),
               cex = 1.3, font=2, family="", pos=4, col="darkblue")
          text(x.min, 1.1*y.max - 0.06*y.len, paste("IC:", signif(IC, 3), " p-val:", p.val), cex = 1.1, font=2,
               family="", pos=4, col="darkgreen")

          for (i in 1:length(xp.projection)) {   # Missing value
             if (feature.vec[i] == max.cont.color) {
                if (show_missing_samples  == T) points(xp.projection[i], yp.projection[i], col="white", pch=4, cex=0.2)
             } else { # Present value
                points(xp.projection[i], yp.projection[i], col="black", bg=col.points[i], pch=21, cex=point.cex)
                if (plot_sample_names == T) {
                        pointLabel(xp.projection[i], yp.projection[i],
                            labels=colnames(m.1.projection)[i], cex=cex_sample_names, col="darkgreen")
                    }
            }
           }

          for (i in 1:nrow(row.objects)) {
             pos <- ifelse(row.objects[i,2] < 0.5, 1, 3)
             text(row.objects[i,1], row.objects[i,2], labels = node.nicknames[i], col="darkblue", cex=1.35, pos=pos, offset=1)        
          }
          points(row.objects[,1], row.objects[,2], col="darkblue", bg="darkblue", pch=21, cex=point.cex)

           
      # Legend

       locs <- !is.na(feature.vec2)
       feature.vec3 <- feature.vec2[locs]
       min.val <- signif(min(feature.vec3), 2)
       max.val <- signif(max(feature.vec3), 2)
       mid.val <- signif(min(feature.vec3) + 0.5*(max(feature.vec3) - min(feature.vec3)), 2)
 
       if (produce.boxplots == T) {
   
           predicted.state.test3 <- match(predicted.state.test[locs], all.classes)
           if (sd(feature.vec3) == 0) {
               feature.vec4 <- y.max*1.05 - 0.02*y.len + 0.5*y.len/6 + rep(0, length(feature.vec3))
           } else {
                 feature.vec4 <-  y.max*1.05 - 0.06*y.len +                
                               (y.len/5.5)*(feature.vec3 - min(feature.vec3))/(max(feature.vec3) - min(feature.vec3))
           }
           classes <- sort(unique(predicted.state.test3))
           boxplot(feature.vec4 ~ predicted.state.test3, main="", boxwex = 0.04*5/k.classes,
                    at = x.max - 0.38*x.len + (x.len/2.75)*classes/k.classes, names = classes,
                    col = categ.col2[classes] , axes=F, xlab=NA, ylab=NA, add=T)
            
           points(c(x.max - 0.38*x.len, x.max - 0.38*x.len), c(y.max*1.05 - 0.02*y.len, y.max*1.05 - 0.02*y.len + y.len/7),
                   type="l", lwd=1, col="black")
           points(c(x.max - 0.39*x.len, x.max - 0.38*x.len), c(y.max*1.05 - 0.02*y.len, y.max*1.05 - 0.02*y.len),
                   type="l", lwd=1, col="black")
           points(c(x.max - 0.39*x.len, x.max - 0.38*x.len), c(y.max*1.05 - 0.02*y.len + 0.5*y.len/7,
                           y.max*1.05 - 0.02*y.len + 0.5*y.len/7), type="l", lwd=1, col="black")
           points(c(x.max - 0.39*x.len, x.max - 0.38*x.len), c(y.max*1.05 - 0.02*y.len + y.len/7,
                           y.max*1.05 - 0.02*y.len + y.len/7), type="l", lwd=1, col="black")

           text(x.max - 0.39*x.len, y.max*1.05 - 0.02*y.len, min.val, cex = 0.6, font=2, family="", pos=2, col="black")
           text(x.max - 0.39*x.len, y.max*1.05 - 0.02*y.len + 0.5*y.len/7, mid.val, cex = 0.6, font=2, family="", pos=2, col="black")
           text(x.max - 0.39*x.len, y.max*1.05 - 0.02*y.len + y.len/7, max.val, cex = 0.6, font=2, family="", pos=2, col="black")                 
       }

        legend.x <- x.max - 0.2*x.len        
        legend.y <- y.max - 0.075*y.len

        if (feature.vals.type == "continuous") {
           for (k in 1:20) {
              points(legend.x + k*x.len/100, legend.y - 0.020*y.len, col=mycol[floor(k*512/20)], pch=15, cex=1)
              points(legend.x + k*x.len/100, legend.y - 0.030*y.len, col=mycol[floor(k*512/20)], pch=15, cex=1)
              points(legend.x + k*x.len/100, legend.y - 0.040*y.len, col=mycol[floor(k*512/20)], pch=15, cex=1)              
          }
          points(c(legend.x + x.len/100, legend.x + 20.5*x.len/100), c(legend.y - 0.045*y.len,
                   legend.y - 0.045*y.len), type="l", lwd=1, col="black")
          points(c(legend.x + x.len/100, legend.x + x.len/100), c(legend.y - 0.045*y.len,
                   legend.y - 0.05*y.len), type="l", lwd=1, col="black")
          points(c(legend.x + 10*x.len/100, legend.x + 10*x.len/100), c(legend.y - 0.045*y.len,
                   legend.y - 0.05*y.len), type="l", lwd=1, col="black")
          points(c(legend.x + 20.5*x.len/100, legend.x + 20.5*x.len/100), c(legend.y - 0.045*y.len,
                   legend.y - 0.05*y.len), type="l", lwd=1, col="black")
          
          text(legend.x + x.len/100, legend.y - 0.045*y.len, min.val, cex = 0.6, font=2, family="", pos=1, col="black")
          text(legend.x + 10*x.len/100, legend.y - 0.045*y.len, mid.val, cex = 0.6, font=2, family="", pos=1, col="black")
          text(legend.x + 20.5*x.len/100, legend.y - 0.045*y.len, max.val, cex = 0.6, font=2, family="", pos=1, col="black")   
         
      } else if (feature.vals.type == "categorical") {
         leg.txt <- f.names # f.set
         pch.vec <- rep(21, length(leg.txt))
         legend(x=legend.x, y=legend.y, legend=leg.txt, bty="n", xjust=0, yjust= 1, pch = pch.vec, title="",   
                pt.bg = categ.col2[1:length(f.names)], col = "black", cex = 1, pt.cex = 2.25)
         
      } else if (feature.vals.type == "binary") {
            leg.txt <- c("Absent", "Present")
            legend(x=legend.x, y=legend.y, legend=leg.txt, bty="n", xjust=0, yjust= 1, pch = c(21, 21), title="",
                 pt.bg = c(binary.col[1], binary.col[2]), col = "black", cex = 1, pt.cex = 2.25)
      }
      text(x.min + 0.65*x.len, y.min - 0.02*y.len, "OncoGenic Positional System (Onco-GPS) Map",
           cex= 1, family="Times", fontface="italic", pos=1, font=4, col="darkblue")                 

      } # End loop over features
      flush.console()
      
      } # End loop over feature files

    #  if (!is.null(output.features.file)) {
    #     row.names(all.features) <- all.features.names
    #     ind <- order(all.features["states",], decreasing=F)
    #     all.features <- all.features[, ind]
    #     CCBA_write.gct.v1(gct.data.frame = all.features, descs = all.features.descs, filename = output.features.file)
    #  }

      if (!is.null(output.file)) dev.off()
    }

#-------------------------------------------------------------------------------------------------	
   CCBA_Norm <- function(
   #
   # NORMALIZATION I
   # P. Tamayo Jan 20 2016
   #
    x,
    n)
    {
      x <- (x - min(x))/(max(x) - min(x))
      return((n - 1) * x + 1)
    }

#-------------------------------------------------------------------------------------------------	
   CCBA_Norm2 <- function(
   #
   # NORMALIZATION II
   # P. Tamayo Jan 20 2016
   #
    x,
    n)
   {
      x <- (x - mean(x))/sd(x)
      x[x > 3] <- 3
      x[x < -3] <- -3
      return((n - 1) * (x + 3)/6 + 1)
    }

#-------------------------------------------------------------------------------------------------	
   CCBA_MI.plot <- function(
   #
   # Makes a plot of the different elements of the differential mutual information between two
   # vectors x and y
   # P. Tamayo Jan 20 2016
   #
       x,
       y,
       x.name = quote(x),
       y.name = quote(y),
       delta = c(bcv(x), bcv(y)),
       n.grid=50)
    {

      x.set <- !is.na(x)
      y.set <- !is.na(y)
      overlap <- x.set & y.set

      x <- x[overlap] +  0.000000001*runif(length(overlap))
      y <- y[overlap] +  0.000000001*runif(length(overlap))

#         delta = c(bcv(x), bcv(y))
         rho <- cor(x, y)
         rho2 <- abs(rho)
         delta <- delta*(1 + (-0.75)*rho2)
         kde2d.xy <- kde2d(x, y, n = n.grid, h = delta)
         FXY <- kde2d.xy$z + .Machine$double.eps
         dx <- kde2d.xy$x[2] - kde2d.xy$x[1]
         dy <- kde2d.xy$y[2] - kde2d.xy$y[1]
         PXY <- FXY/(sum(FXY)*dx*dy)
         PX <- rowSums(PXY)*dy
         PY <- colSums(PXY)*dx
         HXY <- -sum(PXY * log(PXY))*dx*dy
         HX <- -sum(PX * log(PX))*dx
         HY <- -sum(PY * log(PY))*dy
         PX <- matrix(PX, nrow=n.grid, ncol=n.grid)
         PY <- matrix(PY, byrow = TRUE, nrow=n.grid, ncol=n.grid)
         MI <- sum(PXY * log(PXY/(PX*PY)))*dx*dy
         IC <- sign(rho) * sqrt(1 - exp(- 2 * MI))

   mycol <- vector(length=512, mode = "numeric")
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), g511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)

   nf <- layout(matrix(c(1,2,3,4,5,6,7,8), 2, 4, byrow=T), c(1,1,1,1), c(1,1), TRUE)   

   plot(x, y, pch=20, xlab=x.name, ylab=y.name, col="black", cex=2)
   
#   PX <- matrix(PX, nrow=n.grid, ncol=n.grid)
#   PY <- matrix(PY, byrow = TRUE, nrow=n.grid, ncol=n.grid)
#   print(log2(PXY/(PX*PY)))
   MIXY <- PXY * log2(PXY/(PX*PY))
   MIXY2 <- PXY * (log2(PXY/(PX*PY)))^2

   PXn <- CCBA_Norm2(PX, ncolors)
   PYn <- CCBA_Norm2(PY, ncolors)
   PXYn <- CCBA_Norm2(PXY, ncolors)
   PX.PYn <- CCBA_Norm2(PX * PY, ncolors)
   PXY_PX.PYn <- CCBA_Norm2(PXY/(PX * PY), ncolors)
   LogPXY_PX.PYn <- CCBA_Norm2(log2(PXY/(PX * PY)), ncolors)
   MIXYn <- CCBA_Norm2(MIXY, ncolors)
   
   image(PXn, main = paste("P(X=", x.name, ")"), zlim = c(0, ncolors), col=mycol, cex.main=1)
   image(PYn, main = paste("P(Y=", y.name, ")"), zlim = c(0, ncolors), col=mycol, cex.main=1)

   image(PXYn, main = paste("P(X, Y)"), col=mycol, zlim = c(0, ncolors), cex.main=1)
   image(PX.PYn, main = paste("P(X)P(Y)"), col=mycol, zlim = c(0, ncolors), cex.main=1)
   image(PXY_PX.PYn, main = paste("P(X, Y)/P(X)P(Y)"), col=mycol, zlim = c(0, ncolors), cex.main=1)
   image(LogPXY_PX.PYn, main = paste("Log2 P(X, Y)/P(X)P(Y)"), col=mycol, zlim = c(0, ncolors), cex.main=1)
   image(MIXYn, main = paste("P(X, Y) Log2 P(X, Y)/P(X)P(Y)    MI:", signif(MI, 3)),
         sub=paste("IC=", signif(IC, 3)), col=mycol, zlim = c(0, ncolors), cex.main=1)

 }

#-------------------------------------------------------------------------------------------------	
   CCBA_train_Bayesian_predictor <- function(
   #
   # Makes a Bayesian predictor for continuous (via discretization), binary or multi-class targets
   # P. Tamayo Feb 20 2016
   #
      target.dataset,                      # Target dataset
      target,                              # Name of target (a row of target.dataset)
      features.dataset,                    # Features dataset
      features,                            # Set of features (rows of features.dataset) to use as input to model
      output.plots,                        # File with output plots
      ev.file             = NULL,          # Output weight of evidence file
      normalization       = "standardize", # Normalization for features
      discretize.target   = F,             # Discretize target (for continuous targets)
      model.file,                          # Bayesian model file (to be use in a test set using the "apply" function
      output.dataset.gct,                  # Output dataset (train set predictions) as a TXT file
      output.dataset.txt)                  # Output dataset (train set predictions) as a GCT file

{

      pdf(file=output.plots, height=8.5, width=11)

      m.1 <- CCBA_read_GCT_file.v1(filename = target.dataset)
      m.1 <- m.1$ds
      target.col.names <- colnames(m.1)
      m.2 <- CCBA_read_GCT_file.v1(filename = features.dataset)
      m.2 <- data.matrix(m.2$ds)

      target.train.vec <- as.character(m.1[target,])

     ind <- order(target.train.vec, decreasing=F)
     target.train.vec <- target.train.vec[ind]
     target.col.names <- target.col.names[ind]

      if (length(features) > 1) {
         features.train.mat <- m.2[features,]
      } else {
         features.train.mat <- as.matrix(t(m.2[features,]))
      }

      locs <- seq(1, length(target.train.vec))[!is.na(target.train.vec)]
      target.train.vec <- target.train.vec[locs]
      target.col.names <-  target.col.names[locs]

      overlap <- intersect(target.col.names, colnames(features.train.mat))
      locs1 <- match(overlap, target.col.names)
      locs2 <- match(overlap, colnames(features.train.mat))
      target.train.vec <- target.train.vec[locs1]
      if (length(features) > 1) {
         features.train.mat <- features.train.mat[, locs2]
      } else {
         features.train.mat <- as.matrix(t(features.train.mat[, locs2]))
         row.names(features.train.mat) <- features.train
      }

     # Normalize features

     if (normalization == "standardize") {
        for (k in 1:length(features)) {
           if (length(unique(features.train.mat[k,])) > 2) {
              row.mean <- mean(features.train.mat[k,])
              row.sd <- sd(features.train.mat[k,])
              features.train.mat[k,] <- (features.train.mat[k,] - row.mean)/row.sd
              for (j in 1:ncol(features.train.mat)) {
                 if (features.train.mat[k, j] > 3)  features.train.mat[k, j] <- 3
                 if (features.train.mat[k, j] < -3) features.train.mat[k, j] <- -3
              }
            }
         }
    }

    # Discretize target

     target.train.vec0 <- target.train.vec
     if (discretize.target == TRUE) {
         target.train.vec <- (target.train.vec - mean(target.train.vec))/sd(target.train.vec)
         for (i in 1:length(target.train.vec)) target.train.vec[i] <- ifelse(target.train.vec[i] < 0, 1, 0)
         target.classes <- unique(target.train.vec)
         target.classes.num <- length(target.classes)
     } else if (length(unique(target.train.vec)) > 2) { # Categorical target:  make a model list for each class

      # Not supported yet
         
     }
     target.classes <- unique(target.train.vec)
     target.classes.num <- length(target.classes)

     target.train.mat <- prob.model.train <- matrix(0, nrow=target.classes.num, ncol=ncol(features.train.mat), 
                                             dimnames=list(target.classes, colnames(features.train.mat)))
#
     ev.model.train <- array(0, dim = c(target.classes.num, length(features), ncol(features.train.mat)), 
                                dimnames = list(target.classes, features, colnames(features.train.mat)))
#
      for (k in 1:target.classes.num) {
           target.train.mat[k, target.train.vec == target.classes[k]] <- 1
     }
      
   # Make Bayesian model

    model.list<- list(NULL) 
    model.index <-  1
    prior <- vector(length=target.classes.num, mode="numeric")
    for (k in 1:target.classes.num) {
       log.odds <- rep(0, length(target.train.vec))
       prior[k] <- sum(target.train.mat[k,] == 1)/ncol(target.train.mat)
       for (f in 1:length(features)) {
          x <- features.train.mat[features[f],]
          model <- glm(target.train.mat[k,] ~ x,  family=binomial("logit")) 
          prob.model <- predict.glm(object=model, newdata=data.frame(x), type="response")
          model.list[[model.index]] <- model
          ev <- log( (prob.model/(1 - prob.model)) / (prior[k]/(1 - prior[k]) ) )
#          
          ev.model.train[k, features[f],] <- ev 
#                         
          log.odds <- log.odds + ev
          model.index <-  model.index + 1
      }
      prob.model.train[k,] <- exp(log.odds)/(exp(log.odds) + 1)
  }

  if (!is.null(ev.file)) {
     print("Evidence array")
     print(ev.model.train[, 1:3, 1:5])

     for (f in 1:length(features)) {
        append <- ifelse(f == 1, F, T)
        tab <- ev.model.train[, features[f],]
        write(noquote(features[f]), file = ev.file, append = append, ncolumns = 1)     
        col.names <- paste(colnames(features.train.mat), collapse = "\t")
        col.names <- paste("Class", col.names, sep= "\t")
        write(noquote(col.names), file = ev.file, append = T, ncolumns = length(col.names))
        write.table(tab, file=ev.file, quote=F, col.names = F, row.names = T, append = T, sep="\t")
      }
   }
      
   predicted.model <- vector(length=ncol(target.train.mat), mode="character")
   for (i in 1:ncol(target.train.mat)) {
        predicted.model[i] <-   target.classes[which.max(prob.model.train[,i])]
   }

  print(prob.model.train)
   for (i in 1:ncol(target.train.mat)) {
      prob.sum <- sum(prob.model.train[, i])
      prob.model.train[, i] <- prob.model.train[, i]/prob.sum
   }
  print(prob.model.train)


  # Brier score

      Brier_score <- vector(length=length(predicted.model), mode = "numeric")
      for (i in 1:ncol(target.train.mat)) {
         Brier_score[i] <- 0
         for (k in 1:target.classes.num) {
             if (predicted.model[i] == target.classes[k]) {
                  Brier_score[i] <- Brier_score[i] + (1 - prob.model.train[k, i])^2
             } else {
                  Brier_score[i] <- Brier_score[i] + prob.model.train[k, i]^2
             }
         }
        Brier_score[i] <- signif(1 - Brier_score[i], digits=2)
       }
      Brier <- signif(mean(Brier_score), digits=2)

      print(Brier_score)
      print(Brier)

   results <- noquote(rbind(target.train.vec, prob.model.train,   predicted.model, Brier_score))
   CCBA_write.gct.v1(gct.data.frame = results, descs = row.names(results), filename = output.dataset.gct)

   results <- t(results)
   col.names <- paste(colnames(results), collapse = "\t")
   col.names <- paste("SAMPLE", col.names, sep= "\t")
   write(noquote(col.names), file = output.dataset.txt, append = F, ncolumns = length(col.names))
   write.table(results, file=output.dataset.txt, quote=F, col.names = F, row.names = T, append = T, sep="\t")

   print(results)

   c.table.model <- table(predicted.model, target.train.vec, dnn= c("Predicted", "Actual"))
   print(c.table.model)
   library(gmodels)
   CT <- CrossTable(c.table.model, chisq=T)
   error <- ifelse(predicted.model == target.train.vec, 0, 1)
   terror <- sum(error)
   perror <- signif(terror/length(error), 3)
   print(paste("Total error (count):", terror))
   print(paste("Percent error:", perror))

   if (target.classes.num == 2) {
      perf.auc.model <- roc.area(target.train.vec, prob.model.train[2,])
      roc.plot(target.train.vec, prob.model.train[2,])
      auc.roc.model <- signif(perf.auc.model$A, digits=3)
      auc.roc.p.model <- signif(perf.auc.model$p.value, digits=3)
      print(paste("AUC ROC Train Model (model fit):", auc.roc.model))
      print(paste("p-val Train Model (model fit):", auc.roc.p.model))
   }

  # Plot

      phen.col <-   c("plum3", "steelblue2", "seagreen3", "orange", "indianred3",  "cyan3", brewer.pal(7, "Set1"), brewer.pal(7,"Dark2"), 
                            brewer.pal(7, "Set1"),  brewer.pal(7, "Paired"), brewer.pal(8, "Accent"), brewer.pal(8, "Set2"),
                            brewer.pal(11, "Spectral"), brewer.pal(12, "Set3"))

      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Purples")[9])) 
      mycol <- colfunc(100)
      colfunc2 <- colorRampPalette(c("white", brewer.pal(9, "Greens")[9])) 
      mycol2 <- colfunc2(100)

   nf <- layout(matrix(c(1, 2, 3, 4, 5, 6), nrow=6, ncol=1, byrow=T),1, c(1.25, 1.25, target.classes.num - 1, 1.25, 1.25, 3), FALSE)

   V1.phen <- match(target.train.vec, target.classes)
   left.margin <- 20
   par(mar = c(1, left.margin, 2, 6))
   image(1:length(V1.phen), 1:1, as.matrix(V1.phen), col=phen.col[1:max(V1.phen)], axes=FALSE, main="True Class", sub = "", xlab= "", ylab="")
#   axis(2, at=1:1, labels="True Class", adj= 0.5, tick=FALSE, las =   1,cex.axis=1, font.axis=1, line=-1)

      boundaries <- NULL
      for (i in 2:length(target.train.vec)) {
         if (target.train.vec[i] != target.train.vec[i-1]) boundaries <- c(boundaries, i-1)
      }
      boundaries <- c(boundaries, length(target.train.vec))
      locs.bound <- c(boundaries[1]/2, boundaries[2:length(boundaries)]
                   - (boundaries[2:length(boundaries)] - boundaries[1:(length(boundaries)-1)])/2)
   text(locs.bound + 0.5, 1, labels=target.classes, adj=c(0.25, 0.5), srt=0, cex=1.5)

   V2.phen <- match(predicted.model, target.classes)
   left.margin <- 20

   r.text <- paste("(count:", terror, " percent:", perror, ")")
   image(1:length(V2.phen), 1:1, as.matrix(V2.phen), col=phen.col[1:max(V1.phen)], axes=FALSE, main="Predicted Class",
                           xlab= "", ylab="", sub="")
#   axis(2, at=1:1, labels="Predicted Class", adj= 0.5, tick=FALSE, las = 1, cex.axis=1, font.axis=1, line=-1)

   H <- prob.model.train
   V2 <- apply(H, MARGIN=2, FUN=rev)
   lower.space <-  ceiling(4 + 100/nrow(H))
#   par(mar = c(lower.space, left.margin, 2, 6))
   image(1:ncol(V2), 1:nrow(V2), t(V2), col=mycol, zlim=c(0,1), axes=FALSE,main="Predicted Class Probabilities",  sub = "", xlab= "",  ylab="") 
    cex.rows <- 0.40 + 200/(nrow(V2) * max(nchar(row.names(V2))) + 200)
   axis(2, at=1:nrow(V2), labels=row.names(V2), adj= 0.5, tick=FALSE, las = 1, cex.axis=cex.rows, font.axis=1, line=-1)

   image(1:length(Brier_score), 1:1, as.matrix(Brier_score),col=mycol2, zlim=c(0,1),  axes=FALSE, main=paste("Brier Confidence Score (",   Brier, ")", sep=""),   sub = "", xlab= "",ylab="")
   image(1:length(error), 1:1, as.matrix(error), col=c("white", "red"), zlim=c(0,1),  axes=FALSE, main=paste("Error",  r.text), sub = "", xlab= "", ylab="")
   cex.cols <- 0.20 + 200/(ncol(V2) * max(nchar(colnames(V2))) + 200)
   mtext(colnames(V2), at=1:ncol(V2), side = 1, cex=cex.cols, col=phen.col[V1.phen], line=0, las=3, font=2, family="")

# SVM predictor

    library(e1071)
    svm.model <- svm(x = t(features.train.mat), y = target.train.vec,    type = "C-classification", probability=T)
    predicted.model.svm <- predict(svm.model, newdata = t(features.train.mat), probability=T)

   c.table.model <- table(predicted.model.svm, target.train.vec, dnn= c("Predicted", "Actual"))
   print(c.table.model)
   library(gmodels)
   CT <- CrossTable(c.table.model, chisq=T)

   save(model.list, features, target, normalization, target.classes, target.classes.num, prior, discretize.target, svm.model, file=model.file)

  dev.off()

  }

#-------------------------------------------------------------------------------------------------	
   CCBA_apply_Bayesian_predictor <- function(
   #
   # Applies a previously trained Bayesian predictor for continuous (via discretization), binary or multi-class targets
   # P. Tamayo Feb 20 2016
   #
      target.dataset       = NULL,        # Optional target dataset
      features.dataset,                   # Features dataset
      model.file,                         # Input Bayesian model file 
      output.plots,                       # File with output plots
      output.dataset.gct,                 # Output dataset (test set predictions) as a TXT file
      output.dataset.txt)                 # Output dataset (test set predictions) as a GCT file

 {

     pdf(file=output.plots, height=8.5, width=11)
     load(model.file)

     #  If test dataset targets are known compute performance

     if (!is.null(target.dataset)) {
         m.1 <- CCBA_read_GCT_file.v1(filename = target.dataset)
         m.1 <- m.1$ds
         target.col.names <- colnames(m.1)
      }
      m.2 <- CCBA_read_GCT_file.v1(filename = features.dataset)
      m.2 <- data.matrix(m.2$ds)

     if (!is.null(target.dataset)) {
         target.test.vec <- as.character(m.1[target,])
     }
         if (length(features) > 1) {
            features.test.mat <- m.2[features,]
         } else {
            features.test.mat <- as.matrix(t(m.2[features,]))
      }

     if (!is.null(target.dataset)) {
         locs <- seq(1, length(target.test.vec))[!is.na(target.test.vec)]
         target.test.vec <- target.test.vec[locs]
         target.col.names <-  target.col.names[locs]

         overlap <- intersect(target.col.names, colnames(features.test.mat))
         locs1 <- match(overlap, target.col.names)
         locs2 <- match(overlap, colnames(features.test.mat))
         target.test.vec <- target.test.vec[locs1]

         if (length(features) > 1) {
             features.test.mat <- features.test.mat[, locs2]
         } else {
            features.test.mat <- as.matrix(t(features.test.mat[, locs2]))
            row.names(features.test.mat) <- features.test
         }
      }

     # Normalize features

     if (normalization == "standardize") {
        for (k in 1:length(features)) {
           if (length(unique(features.test.mat[k,])) > 2) {
              row.mean <- mean(features.test.mat[k,])
              row.sd <- sd(features.test.mat[k,])
              features.test.mat[k,] <- (features.test.mat[k,] - row.mean)/row.sd
              for (j in 1:ncol(features.test.mat)) {
                 if (features.test.mat[k, j] > 3)  features.test.mat[k, j] <- 3
                 if (features.test.mat[k, j] < -3) features.test.mat[k, j] <- -3
              }
            }
         }
    }

    # Discretize target

     if (!is.null(target.dataset)) {
        target.test.vec0 <- target.test.vec
        if (discretize.target == TRUE) {
            target.test.vec <- (target.test.vec - mean(target.test.vec))/sd(target.test.vec)
            for (i in 1:length(target.test.vec)) target.test.vec[i] <- ifelse(target.test.vec[i] < 0, 1, 0)

        } else if (length(unique(target.test.vec)) > 2) { # Categorical target:  make a model list for each class

        }
     }
      
   # Apply Bayesian model

    prob.model.test <- matrix(0, nrow=target.classes.num,  ncol=ncol(features.test.mat), 
                               dimnames=list(target.classes, colnames(features.test.mat)))
    model.index <-  1
    for (k in 1:target.classes.num) {
       log.odds <- rep(0, ncol(features.test.mat))
       for (f in 1:length(features)) {
          x <- features.test.mat[features[f],]
          prob.model <- predict.glm(object=model.list[[model.index]], newdata=data.frame(x), type="response")
          ev <- log( (prob.model/(1 - prob.model)) / (prior[k]/(1 - prior[k]) ) )
          log.odds <- log.odds + ev
          model.index <-  model.index + 1
      }
      prob.model.test[k,] <- exp(log.odds)/(exp(log.odds) + 1)
  }

   predicted.model <- vector(length=ncol(features.test.mat), mode="character")
   for (i in 1:ncol(features.test.mat)) {
        predicted.model[i] <-   target.classes[which.max(prob.model.test[,i])]
   }

  print(prob.model.test)
   for (i in 1:ncol(features.test.mat)) {
      prob.sum <- sum(prob.model.test[, i])
      prob.model.test[, i] <- prob.model.test[, i]/prob.sum
   }
  print(prob.model.test)

  ind <- order(predicted.model, decreasing=F)
  predicted.model <- predicted.model[ind]
  prob.model.test <- prob.model.test[, ind]

  if (!is.null(target.dataset)) {
        target.test.vec <- target.test.vec[ind]
  }

  # Brier score

      Brier_score <- vector(length=length(predicted.model), mode = "numeric")
      for (i in 1:ncol(features.test.mat)) {
         Brier_score[i] <- 0
         for (k in 1:target.classes.num) {
             if (predicted.model[i] == target.classes[k]) {
                  Brier_score[i] <- Brier_score[i] + (1 - prob.model.test[k, i])^2
             } else {
                  Brier_score[i] <- Brier_score[i] + prob.model.test[k, i]^2
             }
         }
        Brier_score[i] <- signif(1 - Brier_score[i], digits=2)
       }
      Brier <- signif(mean(Brier_score), digits=2)

      print(Brier_score)
      print(Brier)

   if (!is.null(target.dataset)) {
      error <- ifelse(predicted.model == target.test.vec, 0, 1)
      terror <- sum(error)
      perror <- signif(terror/length(error), 3)
      print(paste("Total error (count):", terror))
      print(paste("Percent error:", perror))
   }

   if (!is.null(target.dataset)) {
        results <- noquote(rbind(target.test.vec, prob.model.test,        predicted.model, Brier_score))
    } else {
        results <- noquote(rbind(prob.model.test, predicted.model, Brier_score))
    }

   CCBA_write.gct.v1(gct.data.frame = results, descs = row.names(results), filename = output.dataset.gct)

   results <- t(results)
   col.names <- paste(colnames(results), collapse = "\t")
   col.names <- paste("SAMPLE", col.names, sep= "\t")
   write(noquote(col.names), file = output.dataset.txt, append = F, ncolumns = length(col.names))
   write.table(results, file=output.dataset.txt, quote=F, col.names = F, row.names = T, append = T, sep="\t")

   print(results)

   if (!is.null(target.dataset)) {
      c.table.model <- table(predicted.model, target.test.vec, dnn= c("Predicted", "Actual"))
      print(c.table.model)
      library(gmodels)
      CT <- CrossTable(c.table.model, chisq=T)

     if (target.classes.num == 2) {
        perf.auc.model <- roc.area(target.test.vec, prob.model.test[2,])
        roc.plot(target.test.vec, prob.model.test[2,])
        auc.roc.model <- signif(perf.auc.model$A, digits=3)
        auc.roc.p.model <- signif(perf.auc.model$p.value, digits=3)
        print(paste("AUC ROC Test Model (model fit):", auc.roc.model))
        print(paste("p-val Test Model (model fit):", auc.roc.p.model))
     }
}

  # Plot
      phen.col <-   c("plum3", "steelblue2", "seagreen3", "orange", "indianred3",  "cyan3", brewer.pal(7, "Set1"), brewer.pal(7,"Dark2"), 
                            brewer.pal(7, "Set1"),  brewer.pal(7, "Paired"), brewer.pal(8, "Accent"), brewer.pal(8, "Set2"),
                            brewer.pal(11, "Spectral"), brewer.pal(12, "Set3"))

      colfunc <- colorRampPalette(c("white", brewer.pal(9, "Purples")[9])) 
      mycol <- colfunc(100)[1:100]
      colfunc2 <- colorRampPalette(c("white", brewer.pal(9, "Greens")[9])) 
      mycol2 <- colfunc2(100)[1:100]

   if (!is.null(target.dataset)) {
      nf <- layout(matrix(c(1, 2, 3, 4, 5, 6), nrow=6, ncol=1, byrow=T),1, c(1.25, 1.25, target.classes.num - 1, 1.25, 1.25, 3), FALSE)
   } else {
      nf <- layout(matrix(c(1, 2, 3, 4), nrow=4, ncol=1, byrow=T),1, c(1.25, target.classes.num - 1, 1.25, 3), FALSE)
   }
   if (!is.null(target.dataset)) {
   V1.phen <- match(target.test.vec, target.classes)
   left.margin <- 20
   par(mar = c(1, left.margin, 2, 6))
   image(1:length(V1.phen), 1:1, as.matrix(V1.phen), col=phen.col[1:max(V1.phen)], axes=FALSE, main="True Class", sub = "", xlab= "", ylab="")

   r.text <- paste("(count:", terror, " percent:", perror, ")")
}

   V1.phen <- match(predicted.model, target.classes)
   left.margin <- 20
   par(mar = c(1, left.margin, 2, 6))

   image(1:length(V1.phen), 1:1, as.matrix(V1.phen),col=phen.col[1:max(V1.phen)], axes=FALSE, main="Predicted Class",sub = "", xlab= "", ylab="")

#   axis(2, at=1:1, labels="Predicted Class", adj= 0.5, tick=FALSE, las = 1, cex.axis=1, font.axis=1, line=-1)

      boundaries <- NULL
      for (i in 2:length(predicted.model)) {
         if (predicted.model[i] != predicted.model[i-1]) boundaries <- c(boundaries, i-1)
      }
      boundaries <- c(boundaries, length(predicted.model))
      locs.bound <- c(boundaries[1]/2, boundaries[2:length(boundaries)]
                   - (boundaries[2:length(boundaries)] -      boundaries[1:(length(boundaries)-1)])/2)

     relevant.classes <- unique(predicted.model)

   text(locs.bound + 0.5, 1, labels=relevant.classes, adj=c(0.25, 0.5), srt=0, cex=1.5)

   H <- prob.model.test
   V2 <- apply(H, MARGIN=2, FUN=rev)
   lower.space <-  ceiling(4 + 100/nrow(H))
#   par(mar = c(lower.space, left.margin, 2, 6))
   image(1:ncol(V2), 1:nrow(V2), t(V2), col=mycol, zlim=c(0,1), axes=FALSE,main="Predicted Class Probabilities",  sub = "", xlab= "",  ylab="") 
    cex.rows <- 0.40 + 200/(nrow(V2) * max(nchar(row.names(V2))) + 200)
   axis(2, at=1:nrow(V2), labels=row.names(V2), adj= 0.5, tick=FALSE, las = 1, cex.axis=cex.rows, font.axis=1, line=-1)

   image(1:length(Brier_score), 1:1, as.matrix(Brier_score),col=mycol2, zlim=c(0,1),  axes=FALSE, main=paste("Brier Confidence Score (", Brier, ")", sep=""), sub = "",xlab= "",ylab="")
   if (!is.null(target.dataset)) {
      image(1:length(error), 1:1, as.matrix(error), col=c("white",   "red"), zlim=c(0,1),  axes=FALSE, main=paste("Error", r.text), sub = "", xlab= "", ylab="")
   }
   cex.cols <- 0.20 + 200/(ncol(V2) * max(nchar(colnames(V2))) + 200)
   mtext(colnames(V2), at=1:ncol(V2), side = 1, cex=cex.cols, col=phen.col[V1.phen], line=0, las=3, font=2, family="")


# SVM predictor

    library(e1071)
    predicted.model.svm <- predict(svm.model, newdata = t(features.test.mat), probability=T)

   if (!is.null(target.dataset)) {
      c.table.model <- table(predicted.model.svm, target.test.vec, dnn= c("Predicted", "Actual"))
      print(c.table.model)
      library(gmodels)

      CT <- CrossTable(c.table.model, chisq=T)
   }


   dev.off()

}     

#-------------------------------------------------------------------------------------------------	

nnls.fit <- function(x,y,wsqrt=1,eps=0,rank.tol=1e-07) {
  ## Purpose: Nonnegative Least Squares (similar to the S-Plus function
  ## with the same name) with the help of the R-library quadprog
  ## ------------------------------------------------------------------------
  ## Attention:
  ## - weights are square roots of usual weights
  ## - the constraint is coefficient>=eps
  ## ------------------------------------------------------------------------
  ## Author: Marcel Wolbers, July 99
  ##
  ##========================================================================
  require ("quadprog")
  m <- NCOL(x)
  if (length(eps)==1) eps <- rep(eps,m)
  x <- x * wsqrt
  y <- y * wsqrt
#  sometimes a rescaling of x and y helps (if solve.QP.compact fails otherwise)
  xscale <- apply(abs(x),2,mean)
  yscale <- mean(abs(y))
  x <- t(t(x)/xscale)
  y <- y/yscale
  Rinv <- backsolve(qr.R(qr(x)),diag(m))
  cf <- solve.QP.compact(Dmat=Rinv,dvec=t(x)%*%y,Amat=rbind(rep(1,m)),
                   Aind=rbind(rep(1,m),1:m),bvec=eps*xscale/yscale,
                         factorized=TRUE)$sol
  cf <- cf*yscale/xscale  #scale back
  cf
}

#-------------------------------------------------------------------------------------------------	

CCBA_read_CLS_file.v1 <- function(file = "NULL") { 
#
# Reads a class vector CLS file and defines phenotype and class labels vectors 
#
      cls.cont <- readLines(file)
      num.lines <- length(cls.cont)
      class.list <- unlist(strsplit(cls.cont[[3]], " "))
      s <- length(class.list)
      t <- table(class.list)
      l <- length(t)
      phen <- vector(length=l, mode="character")
      class.v <- vector(length=s, mode="numeric")
     
      current.label <- class.list[1]
      current.number <- 1
      class.v[1] <- current.number
      phen[1] <- current.label
      phen.count <- 1

      if (length(class.list) > 1) {
         for (i in 2:s) {
             if (class.list[i] == current.label) {
                  class.v[i] <- current.number
             } else {
                  phen.count <- phen.count + 1
                  current.number <- current.number + 1
                  current.label <- class.list[i]
                  phen[phen.count] <- current.label
                  class.v[i] <- current.number
             }
        }
       }
     return(list(phen = phen, class.v = class.v, class.list = class.list))
}

#-------------------------------------------------------------------------------------------------
#
# Generates a file with the association matrix between two input files
# P. Tamayo Feb 2016
#

   CCBA_generate_association_matrix.v1 <- function(
      #
      #  For an input file generate the association matrix between the columns (perturbation/states)
      #
      input_dataset,                    # Input dataset (GCT). This is e.g. an original dataset A or the H matrix
      input_dataset2 = NULL,            # Second input dataset (GCT). This is e.g. an original dataset A or the H matrix                                       
      association_type = "columns",     # Type of association: between "columns" or between "rows"
      exclude_suffix = F,               # Exclude suffix (sub-string after last "_") from row/column names
      annot_file = NULL,                # Column or row annotation file (TXT, optional) in format c(file, name_column, annot_column, use_prefix)
      annot_file2 = NULL,               # Column or row annotation file (TXT, optional) in format c(file, name_column, annot_column, use_prefix)      
      association_metric = "IC",        # Association metric: "IC" (Information Coefficient), "ICR" (IC ranked), "COR" (Pearson), "SPEAR" (Spearman)
      output_assoc_matrix_file,         # Output (GCT) file with association matrix
      output_assoc_plot,                # Output (PDF) file with association plot
      cex.rows = "auto",                # Size of row labels
      cex.cols = "auto",                # Size of col labels
      sort.matrix = T)                  # Sort matrix

   {

   set.seed(5209761)
   
   # Read input dataset

   dataset.1 <- CCBA_read_GCT_file.v1(filename = input_dataset)
   H <- data.matrix(dataset.1$ds)
   print(paste("Dimensions dataset1:", dim(H)))
   if (association_type == "rows") H <- t(H)

   size <- ncol(H)/50
   if (size < 11) size <- 11
   pdf(file=output_assoc_plot, height=size, width=size)

   s <- strsplit(input_dataset, split="/")
   file.name <- s[[1]][length(s[[1]])]

   if (exclude_suffix == T) {
      row.names.H <- vector(length=nrow(H), mode="character")
      for (i in 1:nrow(H)) {
         temp <- unlist(strsplit(row.names(H)[i], split="_"))
         row.names.H[i] <- paste(temp[1:(length(temp)-1)], collaps="_")
     }
   } else {
      row.names.H <- row.names(H)
   }
 
   if (!is.null(input_dataset2)) {
      dataset.2 <- CCBA_read_GCT_file.v1(filename = input_dataset2)
      H2 <- data.matrix(dataset.2$ds)
      print(paste("Dimensions dataset2:", dim(H2)))

     s <- strsplit(input_dataset2, split="/")
     file.name2 <- s[[1]][length(s[[1]])]

      if (association_type == "rows") H2 <- t(H2)

      if (exclude_suffix == T) {
         row.names.H2 <- vector(length=nrow(H2), mode="character")
         for (i in 1:nrow(H2)) {
            temp <- unlist(strsplit(row.names(H2)[i], split="_"))
            row.names.H2[i] <- paste(temp[1:(length(temp)-1)], collaps="_")
         }
      } else {
         row.names.H2 <- row.names(H2)
      }

   } else {
      H2 <- H
      row.names.H2 <- row.names.H
      file.name2 <- file.name
    }
   
   # Read annotation file

   if (!is.null(annot_file)) {
      annot.table <- read.table(annot_file[[1]], header=T, sep="\t", skip=0, colClasses = "character")
      gene.list <- annot.table[, annot_file[[2]]]
      annot.list <- annot.table[, annot_file[[3]]]
      gene.set <- vector(length=ncol(H), mode="character")
      if (annot_file[[4]] == T) {
         for (i in 1:ncol(H)) {
            gene.set[i] <- strsplit(colnames(H)[i], split="_")[[1]]
         }
      } else {
         gene.set <- colnames(H)
      }
      locs <- match(gene.set, gene.list)
      gene.class <- annot.list[locs]
      for (k in 1:length(gene.class)) gene.class[k] <- substr(gene.class[k], 1, 10)
      all.classes <- sort(unique(gene.class))
      colnames(H) <- paste(colnames(H), " (", gene.class, ") ", sep="")
    } else {
      gene.class <- rep(" ", ncol(H))
      all.classes <- " "
   }
   
   if (!is.null(annot_file2)) {
      annot.table2 <- read.table(annot_file2[[1]], header=T, sep="\t", skip=0, colClasses = "character")
      gene.list2 <- annot.table2[, annot_file2[[2]]]
      annot.list2 <- annot.table2[, annot_file2[[3]]]
      gene.set2 <- vector(length=ncol(H2), mode="character")
      if (annot_file2[[4]] == T) {
         for (i in 1:ncol(H2)) {
            gene.set2[i] <- strsplit(colnames(H2)[i], split="_")[[1]]
         }
      } else {
         gene.set2 <- colnames(H2)
      }
      locs <- match(gene.set2, gene.list2)
      gene.class2 <- annot.list2[locs]
      for (k in 1:length(gene.class2)) gene.class2[k] <- substr(gene.class2[k], 1, 10)
      all.classes2 <- sort(unique(gene.class2))
      colnames(H2) <- paste(colnames(H2), " (", gene.class2, ") ", sep="")
    } else if (!is.null(annot_file)) {
       all.classes2 <- all.classes
       gene.class2 <- gene.class
    }

   # Define overlapping set

   overlap <- intersect(row.names.H, row.names.H2)
   print(paste("Size of overlap space:", length(overlap)))
   locs1 <- match(overlap, row.names.H)
   locs2 <- match(overlap, row.names.H2)   
   H <- H[locs1,]
   H2 <- H2[locs2,]

   # Signatures association plot
       
   nf <- layout(matrix(c(1, 2), 2, 1, byrow=T), 1, c(8, 1), FALSE)

   mycol <- vector(length=512, mode = "numeric")   # Red/Blue "pinkogram" color map
   for (k in 1:256) mycol[k] <- rgb(255, k - 1, k - 1, maxColorValue=255)
   for (k in 257:512) mycol[k] <- rgb(511 - (k - 1), 511 - (k - 1), 255, maxColorValue=255)
   mycol <- rev(mycol)
   ncolors <- length(mycol)

   col.classes <- c("darkseagreen2", "mediumorchid2", brewer.pal(7, "Set1"), brewer.pal(7, "Dark2"),
                 brewer.pal(7, "Paired"), brewer.pal(8, "Accent"), brewer.pal(8, "Set2"), brewer.pal(11, "Spectral"), brewer.pal(12, "Set3"),
                 sample(c(brewer.pal(9, "Blues"), brewer.pal(9, "Reds"), brewer.pal(9, "Oranges"), brewer.pal(9, "Greys"),
                          brewer.pal(9, "Purples"), brewer.pal(9, "Greens"))))

   assoc.matrix <- CCBA_compute_assoc_or_dist.v1(input_matrix1 = H, input_matrix2 = H2, object_type = "columns",
                                                     assoc_metric = "IC", distance = F)
   if (sort.matrix == T) {
      dist.matrix <- CCBA_compute_assoc_or_dist.v1(input_matrix1 = assoc.matrix, input_matrix2 = assoc.matrix, object_type = "columns",
                                                     assoc_metric = "IC", distance = T)
      hc <- hclust(dist.matrix, "ward")
      assoc.matrix <- assoc.matrix[, hc$order]
      dist.matrix2 <- CCBA_compute_assoc_or_dist.v1(input_matrix1 = assoc.matrix, input_matrix2 = assoc.matrix, object_type = "rows",
                                                     assoc_metric = "IC", distance = T)
      hc2 <- hclust(dist.matrix2, "ward")
      assoc.matrix <- assoc.matrix[hc2$order,]
  }
   CCBA_write.gct.v1(gct.data.frame = assoc.matrix, descs = row.names(assoc.matrix), filename = output_assoc_matrix_file)
 
   assoc.matrix <- ceiling(ncolors * (assoc.matrix + 1)/2)
   V <- apply(assoc.matrix, MARGIN=2, FUN=rev)
   par(mar = c(3, 10, 9, 4))
   image(1:dim(V)[2], 1:dim(V)[1], t(V), main = "", zlim = c(0, ncolors), col=mycol,
         axes=FALSE, xlab= "", ylab="") 

   mtext("Association Matrix", cex=1.3, side = 3, line = 7, outer=F)   
   mtext(file.name2, cex=1, side = 3, line = 5, outer=F)      
   mtext(file.name, cex=1, side = 2, line = 5, outer=F)

   if (!is.null(annot_file)) {
      cols <- col.classes[match(gene.class[hc$order], all.classes)]
   } else {
      cols <- "black"
   }
   if (!is.null(annot_file2)) {
      cols2 <- col.classes[match(gene.class2[hc2$order], all.classes2)]
   } else {
      cols2 <- cols
   }
   if (cex.rows == "auto") {
       cex.rows <- 0.15 + 180/(nrow(V) * max(nchar(row.names(V))) + 200)
       cex.cols <-cex.rows
   }
   mtext(row.names(V), at=1:nrow(V), side = 2, cex=cex.rows, col=rev(cols), line=0, las=1, font=2, family="")
   mtext(colnames(V), at=1:ncol(V), side = 3, cex=cex.cols, col=cols2, line=0, las=3, font=2, family="")

   # Legend

   par(mar = c(3, 25, 1, 5))
   leg.set <- seq(-1, 1, 0.01)
   image(1:length(leg.set), 1:1, as.matrix(leg.set), zlim=c(-1, 1), col=mycol, axes=FALSE, main=paste("Association Metric [", association_metric, "]", sep=""),
       sub = "", xlab= "", ylab="",font=2, family="", mgp = c(0, 0, 0), cex.main=0.8)
   ticks <- seq(-1, 1, 0.25)
   tick.cols <- rep("black", 5)
   tick.lwd <- c(1,1,2,1,1)
   locs <- NULL
   for (k in 1:length(ticks)) locs <- c(locs, which.min(abs(ticks[k] - leg.set)))
   axis(1, at=locs, labels=ticks, adj= 0.5, tick=T, cex=0.6, cex.axis=0.6, line=0, font=2, family="", mgp = c(0.1, 0.1, 0.1))
#   mtext(paste("Association Metric [", association_metric, "]", sep=""), cex=0.3, side = 1, line = 3.5, outer=F)
   
   dev.off()
 }

#-------------------------------------------------------------------------------------------------
# This produces genes x samples GCT files with mutations entries for "All,
# "All_Nonsilent" and "variant class" and "protein changes" entries for
# "maf" formatted mutations files (e.g. TCGA's LUAD-TP.final_analysis_set.maf)

CCBA_produce_mutation_file.v1 <- function(
      maf.mut.input_file,
      gct.output.file,
      format_sample_names = "yes",   # Reformat (TCGA) sample names replacing "-" with"." and truncate to 12 characters
      variant.thres = 3,  # threshold in sample counts to define entries for variant classes (e.g. Missense_Mutation)
      change.thres = 3,  # threshold in sample counts to define entries for protein changes (e.g. p.S768I )
      genes_with_all_entries = NULL, # for this list of genes create entries for all protein changes (disregard change.threshold)
      exclude_flat_features  = F) # exclude flat features
    {
        
   ds <- read.delim(maf.mut.input_file, header=T, row.names = NULL, sep="\t", blank.lines.skip=T,
                    comment.char="", as.is=T)

   if (format_sample_names == "yes") {
      for (i in 1:nrow(ds)) {
          ds[i,"Tumor_Sample_Barcode"] <- substr(paste(strsplit(ds[i,"Tumor_Sample_Barcode"], "-")[[1]], collapse="."), 1, 12)
      }
   } 

   print("ds[i,Tumor_Sample_Barcode]:")   
   print(ds[i,"Tumor_Sample_Barcode"])
   
   gene.list <- ds[,"Hugo_Symbol"]

   N.gene.list <- length(gene.list)
   gene.names <- unique(gene.list)
   N.genes <- length(gene.names)

   sample.names <- ds[,"Tumor_Sample_Barcode"]
   
   u.sample.names <- unique(sample.names)
   N.samples <- length(u.sample.names)

   mut.table <- NULL
   mut.table.row.names <- NULL

   genes_with_all_entries_list <- NULL
   
   for (g in 1:N.genes) {

      gene <- gene.names[g]

      if (g %% 100 == 0) print(paste("processing gene number", g, " out of ", N.genes))
      locs <- seq(1, N.gene.list)[gene.list == gene]
      samples <- ds[locs,"Tumor_Sample_Barcode"]
      variant.classes <- ds[locs,"Variant_Classification"]
      u.variant.classes <- table(variant.classes)
      prot.change.classes<- ds[locs,"Protein_Change"]
      u.prot.change.classes <- table(prot.change.classes)

# Make entry for "All" mutations

      row.entry <- rep(0, N.samples)
      locs3 <- match(samples, u.sample.names)
      row.entry[locs3] <- 1
      mut.table <- rbind(mut.table, row.entry)
      row.entry.count <- sum(row.entry)
      entry.name <- paste(gene, "_MUT_All", sep="")
      mut.table.row.names <- c(mut.table.row.names, entry.name)

      if (!is.na(match(gene, genes_with_all_entries))) {
          print(paste(entry.name, row.entry.count))          
          genes_with_all_entries_list <- c(genes_with_all_entries_list, entry.name)
      }

# Make entry for "Nonsilent" mutations

      locs2 <- seq(1, length(locs))[variant.classes != "Silent"]        
      selected.samples <- samples[locs2]
      row.entry <- rep(0, N.samples)
      locs3 <- match(selected.samples, u.sample.names)
      row.entry[locs3] <- 1
      mut.table <- rbind(mut.table, row.entry)
      row.entry.count <- sum(row.entry)
      entry.name <- paste(gene, "_MUT_Nonsilent", sep="")
      mut.table.row.names <- c(mut.table.row.names, entry.name)

      if (!is.na(match(gene, genes_with_all_entries))) {
          print(paste(entry.name, row.entry.count))
          genes_with_all_entries_list <- c(genes_with_all_entries_list, entry.name)
      }

# Make entries for variant classes 

      variant.classes.above.thres <- names(u.variant.classes)[u.variant.classes >= variant.thres]
      variant.classes.above.thres  <- setdiff(variant.classes.above.thres, "Silent") # exclude Silent mutations
 
      for (var in variant.classes.above.thres) {
         locs2 <- seq(1, length(locs))[variant.classes == var]        
         selected.samples <- samples[locs2]
         row.entry <- rep(0, N.samples)
         locs3 <- match(selected.samples, u.sample.names)
         row.entry[locs3] <- 1
         row.entry.count <- sum(row.entry)
         
         if (exclude_flat_features == T & row.entry.count == 0) next         

         mut.table <- rbind(mut.table, row.entry)         
         entry.name <- paste(gene, "_MUT_", var, sep="")
         mut.table.row.names <- c(mut.table.row.names, entry.name)

         if (!is.na(match(gene, genes_with_all_entries))) {
             print(paste(entry.name, row.entry.count))                       
             genes_with_all_entries_list <- c(genes_with_all_entries_list, entry.name)
         }
         
      } # loop over variant classes

# Make entries for protein change classes 

      if (!is.na(match(gene, genes_with_all_entries))) { # for genes in the list include all protein changes
         print(paste("Making all entries for gene:", gene))
         prot.change.classes.above.thres <- names(u.prot.change.classes)
      } else {
         prot.change.classes.above.thres <- names(u.prot.change.classes)[u.prot.change.classes >= change.thres]
      }

      for (change in prot.change.classes.above.thres) {
         locs2 <- seq(1, length(locs))[prot.change.classes == change]        
         selected.samples <- samples[locs2]
         row.entry <- rep(0, N.samples)
         locs3 <- match(selected.samples, u.sample.names)
         row.entry[locs3] <- 1
         row.entry.count <- sum(row.entry)

         if (exclude_flat_features == T & row.entry.count == 0) next         

         mut.table <- rbind(mut.table, row.entry)
         entry.name <- paste(gene, "_MUT_", change, sep="")
         mut.table.row.names <- c(mut.table.row.names, entry.name)

         if (!is.na(match(gene, genes_with_all_entries))) {
             print(paste(entry.name, row.entry.count))                                    
             genes_with_all_entries_list <- c(genes_with_all_entries_list, entry.name)
         }

     } # loop over prot. change classes

   } # loop over genes

   row.names(mut.table) <-  mut.table.row.names 
   colnames(mut.table) <- u.sample.names

   CCBA_write.gct.v1(gct.data.frame = mut.table, descs =row.names(mut.table), filename = gct.output.file)

   print(paste("Dimensions of input mut table:", N.genes, N.samples))
   print(paste("Dimensions of output mut table:", nrow(mut.table), ncol(mut.table)))

   print(paste(genes_with_all_entries_list, collapse="','"))
   
}

#-------------------------------------------------------------------------------------------------
# This explodes a column in a table and creates binary variables for each categorical values in that column
#
# March 2016

CCBA_explode_columns_in_table <- function(
     phen.table.in,    # Input phenotype table
     phen.table.out,   # Output phenotype table
     gct.file.out = NULL, # optional GCT output file
     phen.columns)     # phenotype columns to explode

{
   samples.table <- read.delim(phen.table.in, header=T, row.names=1, sep="\t", skip=0)
   col.names <- colnames(samples.table)
   dim(samples.table)

   col.count <- ncol(samples.table)
   for (col in phen.columns) {
      phenotypes <- samples.table[, col]
      unique.phenotypes <- unique(phenotypes)
      for (phen in unique.phenotypes) {
         if (phen == "" | (is.na(phen))) next
         print(paste("Adding column:", phen, " from original column:", col))
         vec <- rep(0, nrow(samples.table))
         vec[phenotypes == phen] <- 1
         print(paste("Num samples:", sum(vec)))
         samples.table <- cbind(samples.table, vec)
         col.names <- c(col.names, phen)
       }
    }
   vec <- rep(1, nrow(samples.table))
   samples.table <- cbind(samples.table, vec)
   col.names <- c(col.names, "ALL")

   colnames(samples.table) <- col.names
   col.names
   dim(samples.table)

   write.table(samples.table, file=phen.table.out, quote=F, row.names=T, col.names=NA, sep="\t")
      
   if (!is.null(gct.file.out)) {
      samples.table.t <- t(samples.table)
      CCBA_write.gct.v1(gct.data.frame = samples.table.t, descs = row.names(samples.table.t), filename = gct.file.out)      
   }
}


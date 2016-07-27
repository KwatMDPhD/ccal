#-------------------------------------------------------------------------------------------------------
# This script prepares TCGA (Firehose) files for further analysis 
#
# Pablo Tamayo -- October 30, 2015
#-------------------------------------------------------------------------------------------------------

# TCGA.name <- "HNSC"
# TCGA.name <- "LUAD"
TCGA.name <- "LAML"

# Input files:

mut.file <-               paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "-TP.final_analysis_set.maf", sep="") 
copy.number.file.thres <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/all_thresholded.by_genes.txt", sep="")
copy.number.file.cont <-  paste("~/UCSD_2015/TCGA/", TCGA.name, "/all_data_by_genes.txt", sep="")
gistic.regions.file <-    paste("~/UCSD_2015/TCGA/", TCGA.name, "/all_lesions.conf_99.txt", sep="")
exp.file <-               paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, ".rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt", sep="")
rppa.file <-              paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, ".protein_exp__mda_rppa_core__mdanderson_org__Level_3__protein_normalization__data.data.txt", sep="")
clin.file <-              paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, ".clin.merged.txt", sep="")

genes_with_all_entries <- c("KRAS", "NRAS", "BRAF", "PIK3CA", "CTNNB1", "GNAS")

# Output files:
                    
mut.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_MUT.v1.1.gct", sep="")
cna.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_CNA_thres.v1.gct", sep="")
cna.file3 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_CNA_cont.v1.gct", sep="")
mut.cna.file <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_MUT_CNA.v1.gct", sep="")
exp.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_EXP.v1.gct", sep="")
rppa.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_RPPA.v1.gct", sep="")
clin.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_clin.v1.gct", sep="")
pathways.file2 <- paste("~/UCSD_2015/TCGA/", TCGA.name, "/", TCGA.name, "_PATHWAYS.v1.gct", sep="")

# Load Analysis libraries                       
                    
source("~/UCSD_2015/Analysis/OPAM.library.v8.R")   
source("~/UCSD_2015/Analysis/DISSECTOR_lib.v16.R")
source("~/UCSD_2015/Analysis/FS.library.v10.R")

#-------------------------------------------------------------------------------------------------------
# Read and process mutation data

DISSECTOR_produce_mutation_file.v1(
      maf.mut.input_file = mut.file,
      gct.output.file = mut.file2,
      variant.thres = 3, 
      change.thres = 3,
      genes_with_all_entries = gene_with_all_entries,
      exclude_flat_features = T)
    
#-------------------------------------------------------------------------------------------------------
# Read and process continuous copy number data from GISTIC2

AMP.thres <- 0.25  # define copy number of more than 0.5 TCGA units (2 + 0.5 = 2.5) as amplified
DEL.thres <- -0.25 # define copy number of less than -0.5 (2 - 0.5 = 1.5) as deleted

ds <- read.delim(copy.number.file.cont, header=T, row.names = NULL, sep="\t", blank.lines.skip=T, comment.char="", as.is=T)
cl.names2 <- colnames(ds)
gene.names2 <- ds[,"Gene.Symbol"]
n.genes2 <- length(gene.names2)
cna.table <- data.matrix(ds[,-c(1, 2, 3)])
row.names(cna.table) <- gene.names2

cna.table2 <- matrix(0, nrow=2*nrow(cna.table), ncol=ncol(cna.table))
colnames(cna.table2) <- substr(colnames(cna.table), 1, 12)

one.vector <- rep(1, ncol(cna.table))
zero.vector <- rep(0, ncol(cna.table))

row.names.cna.table <- row.names(cna.table)
row.names.cna.table2 <- NULL

for (i in 1:nrow(cna.table)) {
      cna.table2[2*i - 1,] <- ifelse(cna.table[i,] >= AMP.thres, one.vector, zero.vector)
      cna.table2[2*i,] <- ifelse(cna.table[i,] <= DEL.thres, one.vector, zero.vector)      
      row.names.cna.table2 <- c(row.names.cna.table2, paste(row.names.cna.table[i], "_AMP", sep=""),
                                paste(row.names.cna.table[i], "_DEL", sep=""))
  }
row.names(cna.table2) <- row.names.cna.table2

# Add major MAP and DEL regions from GISTIC

ds <- read.delim(gistic.regions.file, header=T, row.names = NULL, sep="\t", blank.lines.skip=T, comment.char="", as.is=T)

col.names <- colnames(ds)
tcga.samples <- NULL
for (i in 1:ncol(ds)) {
    prefix <- strsplit(col.names[i], "\\.")[[1]][1]
#    print(paste(i, col.names[i], prefix, sep = " | "))
    if (prefix == "TCGA") tcga.samples <- c(tcga.samples, i)
}
    
gistic.table <- row.ids <- NULL

for (i in 1:nrow(ds)) {
   row.elements <- strsplit(ds[i,1], " ")[[1]]
   if (row.elements[length(row.elements)] == "values") next # skip CN values
   suffix <- ifelse (row.elements[1] == "Amplification", "AMP", "DEL")
   region <- ds[i, 2]
   region <- strsplit(region, " ")[[1]][1] # remove trailing spaces
   row.ids <- c(row.ids, paste(region, "_", suffix, sep=""))
   gistic.table <- rbind(gistic.table, ds[i, tcga.samples])
}
   u.row.names <- unique(row.ids)
   locs <- match(u.row.names, row.ids)
   gistic.table <- gistic.table[locs,]
   row.names(gistic.table) <- u.row.names

   gistic.table <- data.matrix(gistic.table)
   gistic.table[gistic.table == 2] <- 1

   colnames(gistic.table) <- substr(colnames(gistic.table), 1, 12)

overlap <- intersect(colnames(gistic.table), colnames(cna.table2))
length(overlap)
locs1 <- match(overlap, colnames(gistic.table))
locs2 <- match(overlap, colnames(cna.table2))

gistic.table <- gistic.table[, locs1]
dim(gistic.table)
cna.table2 <- cna.table2[, locs2]
dim(cna.table2)

cna.table3 <- rbind(gistic.table, cna.table2)
dim(cna.table3)

write.gct.2(gct.data.frame = cna.table3, descs = row.names(cna.table3), filename = cna.file3)

#-------------------------------------------------------------------------------------------------------
# Merge MUT and CNA datasets into one 

  ds <- MSIG.Gct2Frame(filename = mut.file2)
  mut.table <- ds$ds
  ds <- MSIG.Gct2Frame(filename = cna.file3)
  cna.table <- ds$ds

dim(mut.table)
dim(cna.table)
    
overlap <- intersect(colnames(mut.table), colnames(cna.table))
locs1 <- match(overlap, colnames(mut.table))
locs2 <- match(overlap, colnames(cna.table))
length(overlap)

mut.table2 <- mut.table[, locs1]
cna.table2 <- cna.table[, locs2]

mut.cna.table <- rbind(mut.table2, cna.table2)
dim(mut.cna.table)

mut.cna.table[1:30, 1:10]

write.gct.2(gct.data.frame = mut.cna.table, descs = row.names(mut.cna.table), filename = mut.cna.file)

#-------------------------------------------------------------------------------------------------------
# Read and prepare gene expression (RNASeq)

ds <- read.delim(exp.file, header=T, row.names = NULL, sep="\t", blank.lines.skip=T, comment.char="", as.is=T)
g.names <- ds[-1,1]
for (i in 1:length(g.names)) g.names[i] <- strsplit(g.names[i], "\\|")[[1]][1]
exp.table <- data.matrix(ds[-1,-1])
row.names(exp.table) <- g.names
dim (exp.table)

locs <- match(g.names, c("?", "psiTPTE22", "tAKR"))
locs <- seq(1, length(locs))[!is.na(locs)]

exp.table2 <- exp.table[-locs,]
colnames(exp.table2) <- substr(colnames(exp.table2), 1, 12)
dim(exp.table2)

u.gene.names <- unique(row.names(exp.table2))
exp.table2 <- exp.table2[u.gene.names,]
dim(exp.table2)

write.gct.2(gct.data.frame = exp.table2, descs = row.names(exp.file2), filename = exp.file2)

length(intersect(colnames(exp.table2), colnames(mut.cna.table)))

#-------------------------------------------------------------------------------------------------------
# Read and prepare RPPA dataset

ds <- read.delim(rppa.file, header=T, row.names = NULL, sep="\t", blank.lines.skip=T, comment.char="", as.is=T)
p.names <- ds[-1,1]
for (i in 1:length(p.names)) p.names[i] <- strsplit(p.names[i], "\\|")[[1]][1]
rppa.table <- data.matrix(ds[-1,-1])
row.names(rppa.table) <- p.names
dim (rppa.table)
rppa.table2 <- rppa.table
colnames(rppa.table2) <- substr(colnames(rppa.table2), 1, 12)
dim(rppa.table2)

u.p.names <- unique(row.names(rppa.table2))
rppa.table2 <- rppa.table2[u.p.names,]
dim(rppa.table2)

write.gct.2(gct.data.frame = rppa.table2, descs = row.names(rppa.table2), filename = rppa.file2)

length(intersect(colnames(exp.table2), colnames(rppa.table2)))

#-------------------------------------------------------------------------------------------------------
# Read and make file with clinical attributes

ds <- read.delim(clin.file, header=T, row.names = 1, sep="\t", blank.lines.skip=T, comment.char="", as.is=T)

barcodes <- ds["patient.bcr_patient_barcode",] # e.g. tcga-4p-aa8j
barcodes <- toupper(barcodes)
for (i in 1:length(barcodes)) barcodes[i] <- paste(strsplit(toupper(barcodes[i]), "-")[[1]], collapse=".")

# This part is specific to each cancer type because the clinical attributes are unique to each of them

if (TCGA.name == "HNSC") {
    
   HPV.status <- as.character(ds["patient.hpv_test_results.hpv_test_result.hpv_status",])    # positive, negative, indeterminate
   HPV.status <- ifelse(HPV.status == "positive", 1, 0)
   vital.status <- as.character(ds["patient.follow_ups.follow_up.vital_status",])
   vital.status <- ifelse(vital.status == "dead", 1, 0)
   vital.status[is.na(vital.status)] <- 0
   clin.table <- rbind(HPV.status, vital.status)


 } else if (TCGA.name == "LAML") {
   AML.subtype <- as.character(ds["patient.leukemia_french_american_british_morphology_code",])

   molecular.abnorm <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm2 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-2.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm3 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-3.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm4 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-4.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm5 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-5.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm6 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-6.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm7 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-7.molecular_analysis_abnormality_testing_result",])
   molecular.abnorm8 <- as.character(ds["patient.molecular_analysis_abnormality_testing_results.molecular_analysis_abnormality_testing_result_values-8.molecular_analysis_abnormality_testing_result",])
   
   clin.table <- rbind(AML.subtype, molecular.abnorm, molecular.abnorm2, molecular.abnorm3, molecular.abnorm4, molecular.abnorm5, molecular.abnorm6, molecular.abnorm7,
                       molecular.abnorm8)

 } else if (TCGA.name == "LUAD") {                       

 }
   
colnames(clin.table) <- barcodes    
write.gct.2(gct.data.frame = clin.table, descs = row.names(clin.table), filename = clin.file2)


#-------------------------------------------------------------------------------------------------------
# Project expression data into the space of pathways

source("~/UCSD_2015/Analysis/OPAM.library.v8.R")   

   OPAM.project.dataset.7(  
      input.ds                 = exp.file,
      output.ds                = pathways.file2,
      gene.set.databases       = c("~/UCSD_2015/Datasets/Consolidated_gene_sets_database.v1.gmt",
                                   "~/UCSD_2015/HN_PROJECT/HNSCC_datasets/Complementary_signatures.v1.gmt"),
      gene.set.selection       = "ALL",
      sample.norm.type         = "rank",  # "rank", "log" or "log.rank"
      weight                   = 0.75,
      statistic                = "area.under.RES",
      output.score.type        = "ES",
      combine.mode             = "combine.add",  # "combine.off", "combine.replace", "combine.add"
      nperm                    =  1,
      min.overlap              =  1,
      correl.type              =  "rank")             # "rank", "z.score", "symm.rank"

#-------------------------------------------------------------------------------------------------------

### Using the Windows Clipboard, or Passing Data Quickly From Excel to R and Back Again
# http://www.r-bloggers.com/using-the-windows-clipboard-or-passing-data-quickly-from-excel-to-r-and-back-again/
# http://stackoverflow.com/questions/9035674/r-function-to-copy-to-clipboard-on-mac-osx

# Copy data out of R
copy.tbl <- function(obj, size = 4096) {
  sn <- Sys.info()["sysname"]
  if (sn == "Darwin") {
    clip <- paste('clipboard-', size, sep = '')
    f <- pipe("pbcopy", "w")
    write.table(obj, f, row.names = FALSE, sep = '\t')
    close(f)
  } else if (sn == "Windows") {
    clip <- paste('clipboard-', size, sep = '')
    f <- file(description = clip, open = 'w')
    write.table(obj, f, row.names = FALSE, sep = '\t')
    close(f)
  } else {
    stop("Reading from the clipboard is not implemented for your system (",
         sn, ") in this package.")
  }
}


# Paste data into R
paste.tbl <- function() {
  if (sn == "Darwin") {
    f <- pipe("pbpaste")
    df <- read.table(f, sep = '\t', header = TRUE)
    close(f)
    return(df)
  } else if (sn == "Windows") {
    f <- file(description = 'clipboard', open = 'r')
    df <- read.table(f, sep = '\t', header = TRUE)
    close(f)
    return(df)
  } else {
    stop("Reading from the clipboard is not implemented for your system (",
         sn, ") in this package.")
  }
}


# # These all work
# copy.tbl(1:100)
# copy.tbl(letters)
# copy.tbl(my.df)
# copy.tbl(table(my.df$col1))
# copy.tbl(matrix(1:20, nrow = 2))
#
# # If my.df is of moderate size
# copy.tbl(my.df)
#
# # If my.df is huge
# copy.tbl(my.df, 10000)
#
# # Pasting works in a similar way. Select the range in Excel you want to copy
# # (including the header) and press Ctrl+C. Then run
#
# other.df <- paste.table()

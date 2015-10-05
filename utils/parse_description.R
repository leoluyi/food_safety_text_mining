parse_description <- function (url_list) {

  get_des <- function (url) {
    # url <- "https://www.facebook.com/aboutkeelung/photos/a.199819080081200.58708.173996702663438/1078006878929078/?type=1"
    if (is.na(url) | length(url) != 1) return(NA)

    url_parsed <- httr::parse_url(url)
    is_fb_photo <- grepl("facebook.com", url_parsed$hostname) & grepl("\\/photos\\/", url_parsed$path)
    if (length(is_fb_photo)==0) is_fb_photo <- FALSE

    tryCatch(
      if (httr::url_ok(url)) {
        doc <- htmlParse(
          GET(url,
              add_headers(
                `accept`="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                `accept-encoding`="gzip, deflate, sdch",
                `accept-language`="zh-TW,zh;q=0.8,en;q=0.6,zh-CN;q=0.4,ja;q=0.2",
                `upgrade-insecure-requests`="1",
                `user-agent`="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36"
              )),
          encoding = "UTF-8")

        if(is_fb_photo) {
          description <- xpathSApply(doc,
                                     "//div[@id='imagestage']/img[@id='fbPhotoImage']/@src")
        } else {
          description <- xpathSApply(doc,
                                     "/*/head/meta[@property='og:description']/@content")
        }

        description <- unname(description)
        if (is.null(description)) return (NA)
        else description
      } else return (NA)

      ,error=function(e) NA
    )
  }

  unlist(sapply(url_list, get_des, USE.NAMES = FALSE))
}

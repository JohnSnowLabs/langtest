/* Responsive menu
	 ========================================================*/

jQuery(document).ready(function($) {
	jQuery('#responsive_menu').click(function(e) {
      e.preventDefault();
      jQuery(this).toggleClass('close');
      jQuery('.top_navigation').toggleClass('open');
  });
  jQuery('#aside_menu').click(function(e) {
      e.preventDefault();
      jQuery(this).toggleClass('close');
      jQuery('.js-col-aside').toggleClass('open');
      if (jQuery(window).width() <= 1023)
      {
        jQuery('.page__sidebar').toggleClass('open'); 
      }
  });
  jQuery('.toc--ellipsis a').click(function(e) {
    if (jQuery(window).width() <= 767)
      {
        jQuery('.js-col-aside').removeClass('open');
        jQuery('.page__sidebar').removeClass('open');     
        jQuery('#aside_menu').removeClass('close');  
      }       
  });
});

/* Tabs
	 ========================================================*/
   try {
    //Pagination active
    let paginationItems = document.querySelectorAll('.pagination_big li'),
        nextVersionContainer = document.querySelector('#nextver'),
        previosVersionContainer = document.querySelector('#previosver'),
        currentVersionContainer = document.querySelector('#currversion'),
        currentPageTitle = document.querySelector('#section').innerText;
  
    // Set active page and update version containers
    for (let i = 0; i < paginationItems.length; i++) {
      const item = paginationItems[i];
      const itemTitle = item.firstElementChild.innerHTML;
      if (itemTitle === currentPageTitle) {
        item.classList.add('active');
        currentVersionContainer.textContent = itemTitle;       
        if(item.previousElementSibling) {
          previosVersionContainer.textContent = item.previousElementSibling.innerText; 
          previosVersionContainer.parentElement.href = 'release_notes_' + item.previousElementSibling.innerText.replaceAll('.', '_');
        } else {
          previosVersionContainer.parentElement.parentElement.classList.add('hide');
        }
        if(item.nextElementSibling) {
          nextVersionContainer.textContent = item.nextElementSibling.innerText;
          nextVersionContainer.parentElement.href = 'release_notes_' + item.nextElementSibling.innerText.replaceAll('.', '_');
        } else {
          nextVersionContainer.parentElement.parentElement.classList.add('hide');
        }         
        break;
      }
    }
  } catch(e){}

try {
  const tabs = (tabContainer, headerSelector, tabSelector, contentSelector, activeClass, display = 'block') => {
    const tabWrapper = document.querySelectorAll(tabContainer);
    
    tabWrapper.forEach(item => {
    const header = item.querySelector(headerSelector),
          tab = item.querySelectorAll(tabSelector),
          content = item.querySelectorAll(contentSelector);

    function hideTabContent() {
        content.forEach(item => {
            item.style.display = 'none';
        })

        tab.forEach(item => {
            item.classList.remove(activeClass);
        })
    }

    function showTabContent(i = 0) {
        content[i].style.display = display;
        tab[i].classList.add(activeClass);
    }

    header.addEventListener('click', (e) => {
      e.preventDefault();
        const target = e.target;
        if( target &&
            (target.classList.contains(tabSelector.replace(/\./, '')) || 
            target.parentNode.classList.contains(tabSelector.replace(/\./, '')))) {
            tab.forEach((item, i) => {
                if(target == item || target.parentNode == item) {
                    hideTabContent();
                    showTabContent(i);
                }
            })
        }
    });

    hideTabContent();
    showTabContent();
    })
          
  }
  tabs('.tabs-wrapper', '.tabs-header', '.tab-btn', '.tabs-item', 'active');
} catch(e){}
